def editDistance(str1, str2, del_cost=1, ins_cost=1, sub_cost=1):
    n = len(str1)
    m = len(str2)
    D = [[0] * (m + 1) for i in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + del_cost
    for i in range(1, m + 1):
        D[0][i] = D[0][i - 1] + ins_cost
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(
                D[i - 1][j] + del_cost,
                D[i][j - 1] + ins_cost,
                D[i - 1][j - 1] + (sub_cost if str1[i - 1] != str2[j - 1] else 0)
            )
    return D[n][m]

def findClosest(str, data):
    min = float('inf')
    closest = ''
    for word in data:
        distance = editDistance(str, word)
        if distance < min:
            closest = word
            min = distance
    return closest