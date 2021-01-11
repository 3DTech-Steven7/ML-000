学习笔记





如何压榨Cython及OpenMP优化Target Encoding







最近上了王然老师的课，做了当周作业后，有了不少心得，希望能分享一下前段时间的学习成果。





这是一篇纯技术文。








首先我们来看一个问题






假设我们的目标变量是 0，1；0 表示负样本，1 表示 正样本。






所以我们如何拟合模型呢？我们没法表示 1 × 程序员. 所 以我们需要把这些改为一种编码。





假如我们有一套数据集，保安有10%月收入超过1W，程序员有90%月收入超过一月，也就是说我们可以用0.9替换程序员，0.1替换保安来进行预测。


我们的解释变量是一个离散的变量，并且有很多类 （比如说职业）。如果我们每一个样本（x）对应一个职业（y）的话，那我们预测就没有任何意义了，因为你都知道y值了，所以这就是发生了data leakage。
   




这样导致单一样本权重过大，进而高估这个变量对模型的作用，也就是会发生测试的时候很准，在真实情况下，预测率极低的情况。





这类问题就叫Target leakage。






所以，最好的解决办法就是算当前的时候，把当前的值给去掉，我们只算其他样本的平均。














Target encoding





Target encoding 采用 target mean value （among each category） 来给categorical feature做编码。为了减少target variable leak，主流的方法是使用2 levels of cross-validation求出target mean，而我们这里不做复杂的计算，仅仅只是均值。






这是一个一开始写出来的代码，并没有优化过，只是简单实现了功能。





# coding = 'utf-8'
import numpy as np
import pandas as pd


def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result

def main():
    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    result_1 = target_mean_v1(data, 'y', 'x')


if __name__ == '__main__':
    main()


    逻辑是创建一个初始值为0,长度是shape求出pandas的DataFrame其长度的np.nparray后。
    再把遍历一次，算出每个index，并且把除当前的其他index进行groupby操作，去计算平均值和count。

    最后再遍历loc出与data中loc当前的x值是否与groupby相等的均值


    听起来很复杂，也确实很复杂，这个函数是对每个标记计算去除该样本后，按x类型的y值均值，时间复杂度为 O(n2)，并不是最优。






更重要一点就是，用timeit计时发现5000个样本居然要 23.6 秒？？！









所以，现在开始整理一下我们的优化思路。。






最重要的一点就是我们需要先优化python代码（算法复杂度），再去用cython（底层），最后才用并行（多线程多进程），不能本末倒置。
















Python优化






因为colab环境配好了，不需要额外花时间，所以我用的Colab来进行操作。





在我们不使用profiler的情况下，我们只能通过经验和理解去分析排查代码中的Hotspots。





我先尝试自己手动优化一下，至少这样写，把算法复杂度从 O(n2) 降到 O(n)



def target_mean_v2(data:pd.DataFrame, y_name:str, x_name:str) -> np.ndarray:
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result


我们再使用profiler看看，我们Hotspots在什么位置。






目前市面上常见可用的有 cprofile，Vtune，line_profiler





因为cprofile的bug和Vtune的特殊性，毕竟我们不能在arm上干inter不是，所以我们使用line_profiler。






在colab第一次使用，需要pip





pip install line_profiler


一般来说写法是这样的


from line_profiler import LineProfiler
profile = LineProfiler(target_mean_v2) #把函数传递到性能分析器
profile.enable() #开始分析
target_mean_v2(data, 'y', 'x')
profile.disable() #停止分析
profile.print_stats() #打印出性能分析结果


不过我们使用colab可以这样玩



%lprun -f target_mean_v2 target_mean_v2(data, 'y', 'x')


不过首先需要load一下ext


%load_ext line_profiler


可以看到我们瓶颈在遍历的赋予键值对的过程中。







但是即便如此，我们已经优化了100倍的速度了，进入了百毫秒级。






可以看到，我上一版中重复参数很多，我们可以优化一下，将其变成变量。



def target_mean_v3(data:pd.DataFrame, y_name:str, x_name:str) -> np.ndarray:
    data_shape = data.shape[0]

    result = np.zeros(data_shape)
    value_dict = dict()
    count_dict = dict()
    for i in range(data_shape):
        data_loc_x = data.loc[i, x_name]
        data_loc_y = data.loc[i, y_name]
        if data_loc_x not in value_dict:
            value_dict[data_loc_x] = data_loc_y
            count_dict[data_loc_x] = 1
        else:
            value_dict[data_loc_x] += data_loc_y
            count_dict[data_loc_x] += 1
    for i in range(data_shape):
        data_loc_x = data.loc[i, x_name]
        data_loc_y = data.loc[i, y_name]
        result[i] = (value_dict[data_loc_x] - data_loc_y) / (count_dict[data_loc_x] - 1)
    return result


我们又提升了数十毫秒的速度，但是显然不够







这次可以很清晰的看到我们的瓶颈在于pandas的DataFrame去loc的过程。







我尝试换了一种写法




def target_mean_v4(data:pd.DataFrame, y_name:str, x_name:str) -> np.ndarray:
    data_shape = data.shape[0]
    result = np.zeros(data_shape)
    value_dict = dict()
    count_dict = dict()

    x_val_series = data.loc[:, x_name]
    y_val_series = data.loc[:, y_name]
    for i in range(data_shape):
        data_loc_x = x_val_series[i]
        data_loc_y = y_val_series[i]
        if data_loc_x not in value_dict:
            value_dict[data_loc_x] = data_loc_y
            count_dict[data_loc_x] = 1
        else:
            value_dict[data_loc_x] += data_loc_y
            count_dict[data_loc_x] += 1
    for i in range(data_shape):
        data_loc_x = x_val_series[i]
        data_loc_y = y_val_series[i]
        result[i] = (value_dict[data_loc_x] - data_loc_y) / (count_dict[data_loc_x] - 1) 

    return result


比上一版快了一倍，也顺利进入了百毫秒内。








但是我们始终是用pandas去操作，效率并不高，我试图去用values的方法转换成numpy，毕竟numpy比pandas快不少。



def target_mean_v5(data:pd.DataFrame, y_name:str, x_name:str) -> np.ndarray:
    data_shape = data.shape[0]
    result = np.zeros(data_shape)
    value_dict = dict()
    count_dict = dict()

    x_val_series = data.loc[:, x_name].values
    y_val_series = data.loc[:, y_name].values
    for i in range(data_shape):
        data_loc_x = x_val_series[i]
        data_loc_y = y_val_series[i]
        if data_loc_x not in value_dict:
            value_dict[data_loc_x] = data_loc_y
            count_dict[data_loc_x] = 1
        else:
            value_dict[data_loc_x] += data_loc_y
            count_dict[data_loc_x] += 1
    for i in range(data_shape):
        data_loc_x = x_val_series[i]
        data_loc_y = y_val_series[i]
        result[i] = (value_dict[data_loc_x] - data_loc_y) / (count_dict[data_loc_x] - 1) 

    return result


又快了十倍的速度，进入了毫秒级。







可以看到读取data后的变量快了不少。







我们进一步优化，直接不用loc了。


def target_mean_v6(data:pd.DataFrame, y_name:str, x_name:str) -> np.ndarray:
    data_shape = data.shape[0]
    result = np.zeros(data_shape)
    value_dict = dict()
    count_dict = dict()

    x_val_series = data[x_name].values
    y_val_series = data[y_name].values
    for i in range(data_shape):
        data_loc_x = x_val_series[i]
        data_loc_y = y_val_series[i]
        if data_loc_x not in value_dict:
            value_dict[data_loc_x] = data_loc_y
            count_dict[data_loc_x] = 1
        else:
            value_dict[data_loc_x] += data_loc_y
            count_dict[data_loc_x] += 1
    for i in range(data_shape):
        data_loc_x = x_val_series[i]
        data_loc_y = y_val_series[i]
        result[i] = (value_dict[data_loc_x] - data_loc_y) / (count_dict[data_loc_x] - 1) 

    return result


loc转np.ndarray和直接转np.ndarray其实没有太大的差别，但是最重要的是pandas转numpy后效率的大幅度提升。






可以看到在相应的读取快了一些。







这个时候，我觉得python代码在我能力范围内我觉得优化到不错了，我想，是时候进行底层优化了（Cython）













Cython优化







为什么用Cython，这么麻烦干嘛直接用C来写？






    因为你需要解决一个问题，如何将数据传给C，numpy需要做些操作，而做些操作只有在cython中实现，而如果抄下来的话就没有任何意义。



    而在使用cython正常情况下，我们会写到一个后缀名为.pyx的文件中，很遗憾，很多IDE并没有相关的代码提示和代码补全，幸运的是如果你用专业版的Pycharm是有的，而社区版是没有的。


专业版





社区版








然后cython的构建需要一个setup.py的文件，例如：





from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

compile_flags = ['-std=c++11', '-fopenmp']
linker_flags = ['-fopenmp']

module = Extension('hello',
                   ['hello.pyx'],
                   language='c++',
                   include_dirs=[numpy.get_include()], # This helps to create numpy
                   extra_compile_args=compile_flags,
                   extra_link_args=linker_flags)

setup(
    name='hello',
    ext_modules=cythonize(module),
    gdb_debug=True # This is extremely dangerous; Set it to False in production.
)





里面有各种构建的方法，如果要在pyx中使用cimport，还需要把相关库的include加进去，特别是如果后续要使用 openmp，也要加上相应的关键字。





如果想在windows上构建，更多信息的话可以去看官方文档，因为我是在colab上做的，所以没有环境配置的问题。





首先需要使用notebook的cython前需要加载cython





%load_ext Cython





然后在在每个代码栏前加上%%cython，cython使用cdef来定义c变量和c函数，例如





%%cython

cdef:
    int i = 0
    unsigned long j = 1
    signed short k = -3
    long long ll = 1LL
    bint flag = True


接下来，我将变量定义成c变量，以下是numpy的数据类型的变量名







此时因为没有解开GIL的锁，所以暂时没有并行。



%%cython -a

import numpy as np
cimport numpy as cnp
import pandas as pd

def target_mean_v7(data:pd.DataFrame, y_name:str, x_name:str) -> np.ndarray: 
  cdef:
    int data_shape = data.shape[0]
    cnp.ndarray[cnp.float64_t] result = np.zeros(data_shape, dtype=np.float64)
    dict value_dict = {}
    dict count_dict = {}
    cnp.ndarray[cnp.int_t] x_val_array = data[x_name].values
    cnp.ndarray[cnp.int_t] y_val_array = data[y_name].values

  for i in range(data_shape):
    data_loc_x = x_val_array[i]
    data_loc_y = y_val_array[i]
    if data_loc_x not in value_dict:
      value_dict[data_loc_x] = data_loc_y
      count_dict[data_loc_x] = 1
    else:
      value_dict[data_loc_x] += data_loc_y
      count_dict[data_loc_x] += 1
  for i in range(data_shape):
    count = count_dict[x_val_array[i]] - 1
    result[i] = (value_dict[x_val_array[i]] - y_val_array[i]) / count

  return result


     可以看到我们使用cython后，又提升了一个等级的速度，不过遗憾的是我们将不能使用profiler来分析Hotspots了，所以我们之前一定要在python的时候把算法优化到尽可能最好。







 



Cython使用OpenMP并行




   我尝试改进算法，并且将cnp的转换成数组指针，这里我将普通的python函数转换成cython的cpdef函数，并使用prange来代替range进行遍历，这是因为我们要解开Gil的锁，来使用并行进行进一步加速


%%cython -a

import numpy as np
cimport numpy as cnp
import pandas as pd
import cython
cimport cython
from cython.parallel import prange 

cpdef target_mean_v8(data, cnp.str y_name, cnp.str x_name): 
  cdef:
    int data_shape = data.shape[0]
    double[:,] result = np.zeros(data_shape, dtype=np.float64)
    double[:,] value_dict = np.zeros(10, dtype=np.float64)
    double[:,] count_dict = np.zeros(10, dtype=np.float64)
    long[:,] x_val_array = data[x_name].values
    long[:,] y_val_array = data[y_name].values
    int i = 0 

  for i in prange(data_shape, nogil=True):
    value_dict[x_val_array[i]] += y_val_array[i]
    count_dict[x_val_array[i]] += 1
  for i in prange(data_shape, nogil=True):
    result[i] = (value_dict[x_val_array[i]] - y_val_array[i]) / (count_dict[x_val_array[i]] - 1)

  return result


这一次改变的比较大，所以提升很显著，在使用并行后，我们达到了惊人的60微秒。






我用memoryview代替之前的数组指针，并且用关闭越界检查以及关闭包裹来进一步优化代码



%%cython -a

import numpy as np
cimport numpy as cnp
import pandas as pd
import cython
cimport cython
from cython.parallel import prange 

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef target_mean_v9(data, cnp.str y_name, cnp.str x_name): 
  cdef:
    int data_shape = data.shape[0]
    double[::1] result = np.zeros(data_shape, dtype=np.float64)
    double[::1] value_dict = np.zeros(10, dtype=np.float64)
    long[::1] count_dict = np.zeros(10, dtype=np.int64)
    long[::1] x_val_array = np.asfortranarray(data[x_name].values, dtype=np.int64)
    long[::1] y_val_array = np.asfortranarray(data[y_name].values, dtype=np.int64)
    int i = 0 
    long x

  for i in prange(data_shape, nogil=True):
    x = x_val_array[i]
    value_dict[x] += y_val_array[i]
    count_dict[x] += 1
  for i in prange(data_shape, nogil=True):
    x = x_val_array[i]
    result[i] = (value_dict[x] - y_val_array[i]) / (count_dict[x] - 1)

  return result


似乎已经很接近极限速度了，43us。







我尝试在变量的数据类型上下功夫，使用指针和上一版的memoryview进行对比



%%cython -a

import numpy as np
cimport numpy as cnp
import pandas as pd
import cython
cimport cython
from cython.parallel import prange 

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef target_mean_v10(data, cnp.str y_name, cnp.str x_name): 
  cdef:
    int data_shape = data.shape[0]
    double[:,] result = np.zeros(data_shape, dtype=np.float64)
    double[:,] value_dict = np.zeros(10, dtype=np.float64)
    long[:,] count_dict = np.zeros(10, dtype=np.int64)
    long[:,] x_val_array = data[x_name].values
    long[:,] y_val_array = data[y_name].values
    int i = 0 
    long x

  for i in prange(data_shape, nogil=True):
    x = x_val_array[i]
    value_dict[x] += y_val_array[i]
    count_dict[x] += 1
  for i in prange(data_shape, nogil=True):
    x = x_val_array[i]
    result[i] = (value_dict[x] - y_val_array[i]) / (count_dict[x] - 1)

  return result


似乎快了一点，可能指针会更好？







在没有新的思路的情况下，我测试了多种组合，发现可能memoryview与数组指针的混用效率更高？（不确定）



%%cython -a

import numpy as np
cimport numpy as cnp
import pandas as pd
import cython
cimport cython
from cython.parallel import prange 

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef target_mean_v11(data, cnp.str y_name, cnp.str x_name): 
  cdef:
    int data_shape = data.shape[0]
    double[::1] result = np.zeros(data_shape, dtype=np.float64)
    double[::1] value_dict = np.zeros(10, dtype=np.float64)
    long[::1] count_dict = np.zeros(10, dtype=np.int64)
    long[:,] x_val_array = data[x_name].values
    long[:,] y_val_array = data[y_name].values
    long x
    int i = 0 

  for i in prange(data_shape, nogil=True):
    x = x_val_array[i]
    value_dict[x] += y_val_array[i]
    count_dict[x] += 1
  for i in prange(data_shape, nogil=True):
    x = x_val_array[i]
    result[i] = (value_dict[x] - y_val_array[i]) / (count_dict[x] - 1)

  return result


测试结果越来越玄学。。。







我尝试将原先的二维数组输入data 拆分成两个独立的数组 x_val_array 和y_val_array，这样能避免在函数传递指针的过程中出现数组内存不连续的问题。



%%cython -a

import numpy as np
cimport numpy as cnp
import pandas as pd
import cython
cimport cython
from cython.parallel import prange 

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef target_mean_v12(cnp.ndarray[cnp.int_t] x_val_array, cnp.ndarray[cnp.int_t] y_val_array): 
  cdef:
    int data_shape = x_val_array.shape[0]
    double[::1] result = np.zeros(data_shape, dtype=np.float64)
    double[::1] value_dict = np.zeros(10, dtype=np.float64)
    #cnp.ndarray[cnp.float64_t] value_dict = np.zeros(10, dtype=np.float64)
    long[::1] count_dict = np.zeros(10, dtype=np.int64)
    long x
    int i = 0 

  for i in prange(data_shape, nogil=True):
    x = x_val_array[i]
    value_dict[x] += y_val_array[i]
    count_dict[x] += 1
  for i in prange(data_shape, nogil=True):
    x = x_val_array[i]
    result[i] = (value_dict[x] - y_val_array[i]) / (count_dict[x] - 1)

  return result









最后，我们对结果进行正确性测试，发现和初版没有误差，也就是说我们结果一样，速度变快了。





















总结





     可以看出，OpenMP 并行计算加速效果优于纯Cython 加速。不过要提个醒，千万不要提前优化，这样很容易导致代码写不出来，不如先保证正确性，在慢慢去优化。

     在不是算法本身的问题的情况下，去优化代码复杂度，能加不要做指数运算，多个重复参数就变量化，循环能少就少，可以用profiler去分析检测。

    在实现差不多的情况下，可以去考虑细节问题，比如底层实现，在内存中，因为内存地址都是成批读临近数据进 Cache 的，所以数组不连续导致速度变慢，算法的读取方式和数据不一致，没有使用SIMD等各种问题，等细节实现完了再考虑并行。

     讲真，其实还可以优化下去，不过没有太好的思路，之前也考虑过用map，但是不知为何实际测的结果比这array要慢上不少，如果能把两个循环写成一个循环我相信又可以提升不少，而且我并没有去做异常处理，这也是一个过程，不过优化这种事，，是没有尽头的，适可而止，始终保持一颗trade off的心hh
