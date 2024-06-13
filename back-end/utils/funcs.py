from matplotlib import pyplot as plt 

h,w=2025/100,1350/100

# 子图坐标索引
def chunk_list(h,w,ks,skip):
    hi,wi=0,0
    chunk_lisk=[]
    while(hi+ks<=h):
        if(wi+ks<=w):
            tmp=[hi,hi+ks,wi,wi+ks]
            chunk_lisk.append(tmp)
            wi+=skip
        else:
            wi=0
            hi+=skip
    return chunk_lisk

# 打印结果
def print_img(img, path, flag):
    plt.figure(figsize=(13.25,20.5),dpi=100)
    if flag == 'origin':
        plt.imshow(img, cmap=plt.cm.gist_earth)
    elif flag == 'results':
        plt.imshow(img, cmap=plt.cm.gist_stern)
    # plt.colorbar()
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0)
    # plt.show()