import os
import argparse, json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Cifar_2NN, Cifar_CNN, Mnist_2NN, Mnist_CNN, RestNet18
from clients import ClientsGroup, client
from phe import paillier
import tenseal as ts
import time
import random

		

'''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--local_epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batch_size', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
'''
# 中间参数保存路径
def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# torch转list加密
def encrypt_vector(public_key, parameters):
    parameters = parameters.flatten(0).cpu().numpy().tolist()
    parameters = [public_key.encrypt(parameter) for parameter in parameters]
    return parameters

def encrypt_vector1(public_key, parameters):
    parameters = parameters.flatten(0).cpu().numpy().tolist()
    parameters = [public_key.encrypt(parameter) for parameter in parameters]
    return parameters

# list解密
def decrypt_vector(private_key, parameters):
    parameters = [private_key.decrypt(parameter) for parameter in parameters]
    return parameters

def add_noise(parameters, dp, dev): 
    noise = None
    # 不加噪声
    if dp == 0:
        return parameters
    # 拉普拉斯噪声
    elif dp == 1:
        noise = torch.tensor(np.random.laplace(0, sigma, parameters.shape)).to(dev)
    # 高斯噪声
    else:
        noise = torch.cuda.FloatTensor(parameters.shape).normal_(0, sigma)
    
    return parameters.add_(noise)


if __name__=="__main__":

    # 定义解析器
    parser = argparse.ArgumentParser(description='FedAvg')
    parser.add_argument('-c', '--conf', dest='conf')
    arg = parser.parse_args()

    # 解析器解析json文件
    with open(arg.conf, 'r') as f:
        args = json.load(f)

    # 创建中间参数保存目录
    test_mkdir(args['save_path'])

    # 使用gpu or cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device("cpu")


    # 定义使用模型(全连接 or 简单卷积)
    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    elif args['model_name'] == 'cifar_cnn':
        net = Cifar_CNN()
    elif args['model_name'] == 'resnet18':
        net = RestNet18()
    elif args['model_name'] == 'cifar_2nn':
        net = Cifar_2NN()  

    # 如果gpu设备不止一个，并行计算
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    # 定义损失函数和优化器
    loss_func = F.cross_entropy
    opti = optim.Adam(net.parameters(), lr=args['learning_rate'])

    # 定义数据集
    type = args['type']

    # 定义多个参与方，导入训练、测试数据集
    myClients = ClientsGroup(type, args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader
    trainDataLoader = myClients.train_data_loader

    # 每轮迭代的参与方个数
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # 初始化全局参数
    # 属于torch.nn.Module的模型中可学习参数都被包含在该模型的parameters中(可以通过model.parameters(`方法获取)。state_dict其实就是一个字典，
    # 该自点中包含了模型各层和其参数tensor的对应关系。
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    # print(global_parameters)
    # 定义噪声的类型和幅度
    dp = args['noise']
    sigma = args['sigma']

    # 生成密钥
    # public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        # poly_modulus_degree=8192,
        # coeff_mod_bit_sizes=[60, 40, 40, 60]
        poly_modulus_degree=4096,
        coeff_mod_bit_sizes=[20, 20, 19, 19, 20]
        # poly_modulus_degree=8192,
        # coeff_mod_bit_sizes=[40, 30, 30, 30, 30, 50]
    )
    context.generate_galois_keys()
    # context.global_scale = 2 ** 30
    context.global_scale = 2 ** 19

    pre_parameters = None

    #更改平滑系数0.8-0.9
    b = 0.9
    # 全局迭代轮次
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        # context = ts.context(
        #     ts.SCHEME_TYPE.CKKS,
            # poly_modulus_degree=8192,
            # coeff_mod_bit_sizes=[50, 30, 30, 30, 50]
            # poly_modulus_degree=8192,
            # coeff_mod_bit_sizes=[40, 30, 30, 30, 30, 50]
        # )
        # context.generate_galois_keys()
        # context.global_scale = 2 ** 30
        # context.global_scale = 2 ** 40

        # 打乱排序，确定num_in_comm个参与方
        order = np.random.permutation(args['num_of_clients'])
        print(order)


        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        # print(clients_in_comm)

        sum_parameters = None
        # 记录全局参数的shape
        parameters_shape = None

        count = 0
        # numbers = [num_in_comm,num_in_comm-1]
        numbers = [num_in_comm]
        nor = random.choice(numbers)

        print(nor)
        # 可视化进度条对选中参与方local_epoch
        for client in tqdm(clients_in_comm):

            # 本地梯度下降
            local_parameters = myClients.clients_set[client].localUpdate(args['local_epoch'], args['batch_size'], net,
                                                                         loss_func, opti, global_parameters)



            # 初始化sum_parameters
            if sum_parameters is None:
                sum_parameters = {}
                # parameters_shape = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
                    # parameters_shape[key] = var.shape
                    # sum_parameters[key] = add_noise(sum_parameters[key], dp, dev)
                    # print("转为list")
                    # sum_parameters[key] = sum_parameters[key].flatten(0).cpu().numpy().tolist()
                    print("开始加密")
                    # print(sum_parameters[key])
                    sum_parameters[key] = ts.ckks_tensor(context,sum_parameters[key])
                    print("加密完成")
                    # print(sum_parameters[key])

                count = count + 1

            else:
                other_parameters = {}
                # 模拟故障节点

                print("=================")
                print(count)
                print("=================")

                if count != nor:

                    for key, var in local_parameters.items():
                        other_parameters[key] = var.clone()
                        other_parameters[key] = ts.ckks_tensor(context, other_parameters[key])
                        sum_parameters[key] += other_parameters[key]
                    count = count + 1
            # if count == nor:
            #     break

        print(count)
        # print(sum_parameters)
        # 更新全局梯度参数
        # for var in global_parameters:
            # sum_parameters[var] = decrypt_vector(private_key, sum_parameters[var])
            # sum_parameters[var] = torch.reshape(torch.Tensor(sum_parameters[var]), parameters_shape[var])
            # global_parameters[var] = (sum_parameters[var].to(dev) / num_in_comm)

        # for var in sum_parameters:
        #     global_parameters[var] = sum_parameters[var] / torch.tensor(num_in_comm)
        print("=================")
        print(sum_parameters)
        print("=================")

        t = time.perf_counter()
        # 更新全局梯度参数

        if pre_parameters is None:
            pre_parameters = {}
            for var in sum_parameters:
            # print(var.shape)
            # global_parameters[var] = (sum_parameters[var].to(dev) / torch.tensor(num_in_comm))
                global_parameters[var] = (sum_parameters[var] / torch.tensor(count ))

                # global_parameters[var] = (sum_parameters[var] / torch.tensor(num_in_comm))
                pre_parameters[var] = global_parameters[var]
        else:
            for var in sum_parameters:
                # global_parameters[var] = torch.tensor(1-b) * pre_parameters[var] + sum_parameters[var] / torch.tensor(b * num_in_comm)
                print("1======")
                global_parameters[var] = torch.tensor(b/count) * sum_parameters[var]
                # global_parameters[var] = torch.tensor(b/num_in_comm) * sum_parameters[var]
                print("2======")
                global_parameters[var] += torch.tensor(1-b) * pre_parameters[var]
                print("3======")
                pre_parameters[var] = global_parameters[var]
        # print(global_parameters)
        print(f'coast:{time.perf_counter() - t:.8f}s')
        # global_parameters[var] = torch.tensor(1-b)*pre_parameters[var] + sum_parameters[var] / torch.tensor(b*num_in_comm)


        # pre_parameters = {}
        # for var in sum_parameters:
        #     print("1======")
        #     pre_parameters[var] = global_parameters[var]
        #     print("2======")
        #     global_parameters[var] = torch.tensor(b/num_in_comm)*sum_parameters[var]
        #     print("3======")
        #     global_parameters[var] += torch.tensor(1-b)*pre_parameters[var]
        #     print("4======")
        #     pre_parameters[var] = global_parameters[var]
        t = time.perf_counter()
        def decrypt1(enc):
            return enc.decrypt().tolist()
        for var in global_parameters:
            global_parameters[var] = decrypt1(global_parameters[var])
            global_parameters[var] = torch.Tensor(global_parameters[var])
            # global_parameters[var] = torch.reshape(torch.Tensor(global_parameters[var]), parameters_shape[var])
        print(f'dec_coast:{time.perf_counter() - t:.8f}s')
        # print(global_parameters)
        # 不进行计算图构建（无需反向传播）
        with torch.no_grad():
            # 满足评估的条件，用测试集进行数据评估
            if (i + 1) % args['val_freq'] == 0:
                # strict表示key、val严格重合才能执行（false不对齐部分默认初始化）
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                # 遍历每个测试数据
                for data, label in testDataLoader:
                    # 转成gpu数据
                    data, label = data.to(dev), label.to(dev)
                    # 预测（返回结果是概率向量）
                    preds = net(data)
                    # 取最大概率label
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('val_accuracy: {}'.format(sum_accu / num))

                # 遍历每个训练数据
                for data, label in trainDataLoader:
                    # 转成gpu数据
                    data, label = data.to(dev), label.to(dev)
                    # 预测（返回结果是概率向量）
                    preds = net(data)
                    # 取最大概率label
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('train_accuracy: {}'.format(sum_accu / num))

        # 根据格式和给定轮次保存参数信息
        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['local_epoch'],
                                                                                                args['batch_size'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))

