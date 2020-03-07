from inspect import signature
import timeit
import psutil
import multiprocessing
from itertools import product

class FT_Decorator():
    def __init__(self,freq=1,callback=None,show=True):
        self.freq = freq
        self.callback = callback
        self.avgtime = 0
        self.count = 0
        self.show  = show

    def __call__(self,func):
        """
        decorator for printing execution time of a function.
        """
        sig = signature(func)
        def wrapped(*args, **kwargs):
            t1 = timeit.default_timer()
            result = func(*args, **kwargs)
            t2 = timeit.default_timer()


            self.avgtime = (self.count * self.avgtime + t2 - t1 ) / (self.count + 1)
            self.count += 1

            if self.callback:
                self.callback(t2-t1)

            if self.show and (self.count % self.freq == 0):
                print('Run {} {} times: avg: {:.5f}s; latest: {:.5f}s'.format(
                    func.__name__, self.count, self.avgtime, t2-t1))
            return result
        wrapped.__signature__ = sig
        wrapped.__name__ = func.__name__
        return wrapped



def poolMap(task,workload,initializer=None,initargs=None,chunks=None,
            total=None,progress_callback=None,progress_gap=(0,100),**kwargs):
    """
    speed up task in a list by multiprocessing.
    task is the function to apply on workload.
    workload is a iterable containing the task inputs.
    initializerr and initargs can be used to setup the function.
    """
    workerheads=psutil.cpu_count(logical=False)
    worker=multiprocessing.Pool(workerheads,initializer,initargs)
    total = total or len(workload)
    chunksize= int(total//chunks)+1 if chunks else int(total/workerheads/10+1)
    result = []
    count=0
    progress = progress_gap[0]
    for _ in worker.imap(task, workload, chunksize):
      count+=1
      result.append(_)
      if progress_callback :
          current_pro = count/total*(progress_gap[1]-progress_gap[0])+progress_gap[0]
          if current_pro > progress + 1:
              progress = current_pro
              progress_callback(current_pro)
    worker.close()
    worker.join()
    worker.terminate()
    return result


def cartesian(*arrays):
    """
    Give a list of arrays, return iterator of product of elements from arrays.
    similar to python product, but also yield indexes.
    """
    lenths = [len(i) for i in arrays]
    l = product(*lenths)
    p = product(*arrays)
    for i,j in zip(l,p):
        yield i,j
