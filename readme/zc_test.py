from re import T
import torch

a = torch.zeros((4,3,1))






a= torch.arange(30).reshape(5,6)

print('b:',a.repeat(2,1).shape)   # 10 12
print('c:',a.repeat(2,1,1)) # 2 5 6
print(a)
ai=a[:,  None]
print(ai)
print(ai.shape) # 5,6,1

def colorstr(*input):
    """用到下面的check_git_status、check_requirements等函数  train.py、val.py、detect.py等文件中
    把输出的开头和结尾加上颜色  命令行输出显示会更加好看  如: colorstr('blue', 'hello world')
    Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code
    """
    # 如果输入长度为1, 就是没有选择颜色 则选择默认颜色设置 blue + bold
    # args: 输入的颜色序列 string: 输入的字符串
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    # 定义一些基础的颜色 和 字体设置
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    # 把输出的开头和结尾加上颜色  命令行输出显示会更加好看
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
tinydict = {'a': 1, 'b': 2, 'b': '3'}

print(colorstr('xxx:') +', '.join(f'{k}={v}' for k, v in tinydict.items())) 

# logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

output = [torch.zeros((0, 6))] * 2

print(output)



a =not False and not False
print(a)