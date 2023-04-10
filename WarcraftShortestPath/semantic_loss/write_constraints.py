import os

def valid_coordinates(x, y):
    if x >= 0 and x <= 11:
        if y >= 0 and y <= 11:
            return True
    return False

file_name = "constraints_generated.txt"

matrix = [
    ["X0.0", "X1.0", "X2.0", "X3.0", "X4.0", "X5.0", "X6.0", "X7.0", "X8.0", "X9.0", "X10.0", "X11.0"],
    ["X0.1", "X1.1", "X2.1", "X3.1", "X4.1", "X5.1", "X6.1", "X7.1", "X8.1", "X9.1", "X10.1", "X11.1"],
    ["X0.2", "X1.2", "X2.2", "X3.2", "X4.2", "X5.2", "X6.2", "X7.2", "X8.2", "X9.2", "X10.2", "X11.2"],
    ["X0.3", "X1.3", "X2.3", "X3.3", "X4.3", "X5.3", "X6.3", "X7.3", "X8.3", "X9.3", "X10.3", "X11.3"],
    ["X0.4", "X1.4", "X2.4", "X3.4", "X4.4", "X5.4", "X6.4", "X7.4", "X8.4", "X9.4", "X10.4", "X11.4"],
    ["X0.5", "X1.5", "X2.5", "X3.5", "X4.5", "X5.5", "X6.5", "X7.5", "X8.5", "X9.5", "X10.5", "X11.5"],
    ["X0.6", "X1.6", "X2.6", "X3.6", "X4.6", "X5.6", "X6.6", "X7.6", "X8.6", "X9.6", "X10.6", "X11.6"],
    ["X0.7", "X1.7", "X2.7", "X3.7", "X4.7", "X5.7", "X6.7", "X7.7", "X8.7", "X9.7", "X10.7", "X11.7"],
    ["X0.8", "X1.8", "X2.8", "X3.8", "X4.8", "X5.8", "X6.8", "X7.8", "X8.8", "X9.8", "X10.8", "X11.8"],
    ["X0.9", "X1.9", "X2.9", "X3.9", "X4.9", "X5.9", "X6.9", "X7.9", "X8.9", "X9.9", "X10.9", "X11.9"],
    ["X0.10", "X1.10", "X2.10", "X3.10", "X4.10", "X5.10", "X6.10", "X7.10", "X8.10", "X9.10", "X10.10", "X11.10"],
    ["X0.11", "X1.11", "X2.11", "X3.11", "X4.11", "X5.11", "X6.11", "X7.11", "X8.11", "X9.11", "X10.11", "X11.11"]]


os.remove(file_name)

f = open(file_name, "a")
f.write("shape [12,12]\n")
f.write("\n")
f.write("# start and end location of the path are always the same\n")
f.write("X0.0 & X11.11\n")
f.write("\n")
f.write("# every tile should have and adjacent tile\n")

for current_y in range(len(matrix)):
    for current_x in range(len(matrix[0])):
        print_string = ""
        print_string += matrix[current_y][current_x]
        print_string += " >> "
        
        possible_moves = [(1,0), (-1,0), (0,1), (0,-1), (-1,-1), (1,-1), (-1,1), (1,1)]
        next_coordinates_list = []
        for move_x, move_y in possible_moves:
            new_x = current_x + move_x
            new_y = current_y + move_y
            if valid_coordinates(new_x, new_y):
                next_coordinates_list.append((new_x, new_y))

        for dropout_index in range(len(next_coordinates_list)):
            print_string += "("
            for index in range(len(next_coordinates_list)):
                if index != dropout_index:
                    print_string += matrix[next_coordinates_list[index][1]][next_coordinates_list[index][0]]
                    print_string += " | "
            print_string = print_string[:-3]
            print_string += ") & "
        print_string = print_string[:-3]

        f.write(print_string)
        f.write("\n")
    f.write("\n")

f.close()