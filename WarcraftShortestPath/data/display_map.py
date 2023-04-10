import matplotlib.pyplot as plt
from copy import deepcopy

def valid_coordinates(nb_rows, nb_columns, x, y):
    if x >= 0 and x < nb_columns and y >= 0 and y < nb_rows:
        return True
    else:
        return False
    
def display_map(map):
    plt.imshow(map)
    plt.show()

def display_map_and_path_boring(map, path):
    map_to_show = deepcopy(map)

    for i in range(len(path)):
        for j in range(len(path[0])):
            if path[i][j] == 1:
                for l in range(8):
                    for k in range(8):
                        map_to_show[(i*8)+l][(j*8)+k] = 0

    plt.imshow(map_to_show)
    plt.show()

def display_map_and_path(map, path):
    map_to_show = deepcopy(map)
    possible_moves = [(1,0), (-1,0), (0,1), (0,-1), (-1,-1), (1,-1), (-1,1), (1,1)]
    previous_i = 0
    previous_j = 0
    constructed_path = [(4,4)]

    while previous_i != len(path) - 1 or previous_j != len(path[0]) - 1:
        for move_i, move_j in possible_moves:
            new_i = previous_i + move_i
            new_j = previous_j + move_j

            if valid_coordinates(len(path), len(path[0]), new_i, new_j):
                if path[new_i][new_j] == 1 and ((new_i * 8) + 4, (new_j * 8) + 4) not in constructed_path:
                    constructed_path.append(((new_i * 8) + 4, (new_j * 8) + 4))
                    break
        
        previous_i = new_i
        previous_j = new_j

    show_x = []
    show_y = []
    for x, y in constructed_path:
        show_x.append(x)
        show_y.append(y)

    plt.plot(show_x, show_y, color="yellow", solid_capstyle='round', linewidth=3)
    plt.imshow(map_to_show)
    plt.axis('off')
    plt.show()