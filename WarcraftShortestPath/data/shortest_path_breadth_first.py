import copy

# class that holds a partial path
class PartialPath():
    def __init__(self, x, y, cost, coordinate_list):
        self.x = x
        self.y = y
        self.cost = cost
        self.coordinate_list = coordinate_list

# checks if the given x and y coordinate are valid given the number of rows and columns
def valid_coordinates(nb_rows, nb_columns, x, y):
    if x >= 0 and x < nb_columns and y >= 0 and y < nb_rows:
        return True
    else:
        return False
    
# check if the given (partial) path holds a loop
def found_loop(path, new_x, new_y):
    for x, y in path.coordinate_list:
        if x == new_x and y == new_y:
            return True
    return False
    
# check if the path ends at the target location (right under)
def end_coordinates(nb_rows, nb_columns, x, y):
    if (x == nb_columns - 1) and (y == nb_rows - 1):
        return True
    else:
        return False
    
# return the shortest path by searching breadth first
# 1 means that the location is part of the shortest path
# 0 means that the location is not part of the shortest path
def find_shortest_path(cost):
    nb_rows = len(cost)
    nb_columns = len(cost[0])
    queue = [PartialPath(0, 0, cost[0][0], [(0, 0)])]
    finished_paths = []

    while len(queue) > 0:
        path = queue.pop(0)
        current_x = path.x
        current_y = path.y
        current_cost = path.cost
        current_coordinate_list = path.coordinate_list

        possible_moves = [(1,0), (-1,0), (0,1), (0,-1), (-1,-1), (1,-1), (-1,1), (1,1)]

        for move_x, move_y in possible_moves:
            new_x = current_x + move_x
            new_y = current_y + move_y
            if valid_coordinates(nb_rows, nb_columns, new_x, new_y):
                if end_coordinates(nb_rows, nb_columns, new_x, new_y):
                    new_cost = current_cost + cost[new_y][new_x]
                    new_coordinate_list = copy.deepcopy(current_coordinate_list)
                    new_coordinate_list.append((new_x, new_y))
                    finished_paths.append(PartialPath(new_x, new_y, new_cost, new_coordinate_list))
                elif not found_loop(path, new_x, new_y):
                    new_cost = current_cost + cost[new_y][new_x]
                    new_coordinate_list = copy.deepcopy(current_coordinate_list)
                    new_coordinate_list.append((new_x, new_y))
                    queue.append(PartialPath(new_x, new_y, new_cost, new_coordinate_list))

    distances = [path.cost for path in finished_paths]
    shortest_path_index = distances.index(min(distances))   
    shortest_path_matrix = [[0 for _ in range(nb_columns)] for _ in range(nb_rows)]

    for x_coordinate, y_coordinate in finished_paths[shortest_path_index].coordinate_list:
        shortest_path_matrix[y_coordinate][x_coordinate] = 1

    return shortest_path_matrix