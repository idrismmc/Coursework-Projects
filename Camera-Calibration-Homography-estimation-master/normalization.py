import numpy as np
def normalization(arr):
    if arr.shape[1]==3:
        xm,ym,zm = np.mean(arr,0)
        std = np.std(arr)
        sq3 = np.sqrt(3)
        u = np.array([[sq3/std,0,0,-sq3*xm/std],
                      [0,sq3/std,0,-sq3*ym/std],
                      [0,0,sq3/std,-sq3*zm/std],
                      [0,0,0,1]])
        
        arr = np.concatenate((arr.T,np.ones((1,arr.shape[0]))))
        normarr = (u@arr)[0:3].T
        return u,normarr
    else:
        xm,ym=np.mean(arr,0)
        std = np.std(arr)
        sq2 = np.sqrt(2)
        u = np.array([[sq2/std,0,-sq2*xm/std],
                      [0,sq2/std,-sq2*ym/std],
                      [0,0,1]])

        arr = arr = np.concatenate((arr.T,np.ones((1,arr.shape[0]))))
        normarr = (u@arr)[0:2].T
        return u,normarr