# check if x and y coordinate are valid given the number of rows and number of columns
def valid_coordinates(nb_rows, nb_columns, x, y):
    if x >= 0 and x < nb_columns and y >= 0 and y < nb_rows:
        return True
    else:
        return False

# find the next node in the graph to examine according the the dijkstra shortest path algorithm
def find_next_coordinates(visited_nodes, working_matrix):
    lowest_cost = None
    next_x = None
    next_y = None
    possible_next_x = None
    possible_next_y = None

    for i in range(len(working_matrix)):
        for j in range(len(working_matrix[i])):
            value = working_matrix[i][j]
            if value is not None:
                cost, _, _ = value
                if (lowest_cost is None) or (cost < lowest_cost):
                    possible_next_y = i
                    possible_next_x = j
                    if not visited_nodes[possible_next_y][possible_next_x]:
                        lowest_cost = cost
                        next_x = possible_next_x
                        next_y = possible_next_y

    return next_x, next_y

# update the values in the dijkstra matrix
def change_cost_in_dijkstra_matrix(working_matrix, cost, current_x, current_y, new_x, new_y):
    cost_to_get_here = working_matrix[current_y][current_x][0]
    if (working_matrix[new_y][new_x] == None) or ((cost[new_y][new_x] + cost_to_get_here) < working_matrix[new_y][new_x][0]):
        working_matrix[new_y][new_x] = (cost[new_y][new_x] + cost_to_get_here, current_x, current_y)    

# return a matrix with the same size as the given cost matrix
# 1 indicate that the location is part of the shortest path
def find_shortest_path(cost):
    nb_rows = len(cost)
    nb_columns = len(cost[0])
    current_x = 0
    current_y = 0
    
    visited_nodes = [[False for _ in range(nb_columns)] for _ in range(nb_rows)]
    dijkstra_matrix = [[None for _ in range(nb_columns)] for _ in range(nb_rows)]
    dijkstra_matrix[0][0] = (cost[0][0], 0, 0)

    while False in [item for row in visited_nodes for item in row]:
        visited_nodes[current_y][current_x] = True
        possible_moves = [(1,0), (-1,0), (0,1), (0,-1), (-1,-1), (1,-1), (-1,1), (1,1)]

        for move_x, move_y in possible_moves:
            new_x = current_x + move_x
            new_y = current_y + move_y
            if valid_coordinates(nb_rows, nb_columns, new_x, new_y):
                change_cost_in_dijkstra_matrix(dijkstra_matrix, cost, current_x, current_y, new_x, new_y)

        current_x, current_y = find_next_coordinates(visited_nodes, dijkstra_matrix)

    complete_path = [[0 for i in range(nb_rows)] for j in range(nb_columns)]
    complete_path[nb_rows-1][nb_columns-1] = 1
    searching_x = nb_rows - 1
    searching_y = nb_columns - 1

    while searching_x != 0 or searching_y != 0:
        complete_path[searching_y][searching_x] = 1
        _, searching_x, searching_y = dijkstra_matrix[searching_y][searching_x]

    complete_path[searching_y][searching_x] = 1
    return complete_path