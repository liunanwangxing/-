import heapq


def find_top_n_numbers_with_indices(lst, n):
    # 找到前N个最大的数及其对应的下标
    if n > len(lst):
        print("N is greater than the length of the list.")
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

    return largest_n, indices


# 示例列表
numbers = [10, 20, 5, 30, 8, 50, 12, 60, 90, 100, 25, 70, 85, 40, 45]

# 找到前N个最大的数及其下标
N = 5
largest_numbers, indices = find_top_n_numbers_with_indices(numbers, N)

print("Largest numbers:", largest_numbers)
print("Indices:", indices)
