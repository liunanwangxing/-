import heapq
#由于过慢，尚待优化
def find_top_n_numbers_with_indices(lst, n):
    # 找到前N个最大的数及其对应的下标
    if n > len(lst):
        return [], []

    # 使用heapq.nlargest找到前N个最大的数
    largest_n = heapq.nlargest(n, lst)

    # 获取这些数的下标
    indices = []
    for num in largest_n:
        index = lst.index(num)
        indices.append(index)
        # 将已找到的元素置为一个不可能的值以防止重复
        lst[index] = float('-inf')

    return indices
def close(img,x,y,w,h):
    global img_weighting
    values = []
    locations = []
    sum_point = 0
    for i in range(w):
        for j in range(h):
            if img[y+j,x+i] == 1:
                continue
            sum = 0
            sum_point+=1
            for k in range(-20,21):
                for l in range(-20,21):
                    if k*k + l*l >400:
                        continue
                    if img[y+j+l,x+i+k] == 0:
                        continue
                    gr = k*k + l*l
                    if gr<=10:
                        mr = 5
                    elif gr>=11 and gr<= 15:
                        mr = 5
                    else:
                        mr = 5
                    sum+=mr
            values.append(sum)
            locations.append(Point(y+j,x+i))
    indices = find_top_n_numbers_with_indices(values,int(0.03*sum_point))
    for m in range(len(indices)):
        point_x = locations[indices[m]].x
        point_y = locations[indices[m]].y
        img_weighting[point_y,point_x] = 1
