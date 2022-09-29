def data_to_result(data):

    dict_result = {}
    new_keys = []
    new_values = []

    keys = list(set(data.values()))
    for i in keys:
        new_keys.insert(0,i)

    values = list(data.keys())
    item_1 = [values[0], values[-1]]
    item_2 = values[1:-1]

    new_values.append(item_1)
    new_values.append(item_2)

    lst_zip = zip(new_keys, new_values)

    for (x,y) in lst_zip:
        dict_result[x] = y

    return dict_result

def main():

    data = {"a": "d",
        "b": "e",
        "c": "d"}

    result = data_to_result(data)
    print(result)

if __name__ == '__main__':
    main()
