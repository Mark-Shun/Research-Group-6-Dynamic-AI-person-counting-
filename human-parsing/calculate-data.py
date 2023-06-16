import numpy as np
import os

minumim_parts_index = [2,4,6, 11,14]


def compare_rgb(rgb_table, rgb_to_compare):

    # print("rgb_table",rgb_table, "\n")
    # print("rgb_to_compare",rgb_to_compare, "\n")

    result = np.zeros((rgb_table.shape[0],3 ))
    for i in range(0, rgb_table.shape[0]):
        for j in range(0, 3):
            result[i][j] = abs((rgb_table[i][j]-rgb_to_compare[i][j])/rgb_to_compare[i][j])
    # print("result", result, "\n")
    return result


def calculate_total(table):
    
    total = 0
    for row_index, row in enumerate(table):
        if row_index in minumim_parts_index:
            total += row[0]
            total += row[1]
            total += row[2]
    return total/(3*len(minumim_parts_index))

def compare_person(person_to_compare, person_to_compare_index, person_average_rgb):
    total_table = np.zeros((99,2))
    total_persons = 0
    average_rgb = np.load("average-rgb/final.npy")
    
    for index, row in enumerate(average_rgb):
        person_bool = row[0][4] == person_to_compare
        if row[0][4] == 5:
            continue
        else:
            if index == person_to_compare_index:
                continue
            person_rgb_table = average_rgb[index]
            total_table[total_persons] = [person_bool,calculate_total(compare_rgb(person_average_rgb, person_rgb_table))]
            total_persons += 1
    return total_table
def main():
    size = 0
    total_everything = np.zeros((9900,2))
    print("Loading: final.npy  ...")
    average_rgb = np.load("average-rgb/final.npy" )
    for index, row in enumerate(average_rgb):
        for i in minumim_parts_index:
            if np.isnan(row[i][0]):
                print(index)
        person = row[0][4]
        person_average_rgb = average_rgb[index]
        compare_person_everything_total = compare_person(person, index, person_average_rgb)
        for i in range(0, 99):
            total_everything[size] = compare_person_everything_total[i]
            if total_everything[size][1] == 0:
                print(index)
            size += 1
        
    np.savetxt('out.txt', total_everything) 
    


if __name__ == '__main__':
    main()


