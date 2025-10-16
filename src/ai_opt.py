from othello_game import OthelloGame
import copy

def get_best_move(game, max_depth=6):
    """
    Given the current game state, this function returns the best move for the AI player using the Alpha-Beta Pruning
    algorithm with a specified maximum search depth.

    Parameters:
        game (OthelloGame): The current game state.
        max_depth (int): The maximum search depth for the Alpha-Beta algorithm.

    Returns:
        tuple: A tuple containing the evaluation value of the best move and the corresponding move (row, col).
    """
    #_, best_move = minmax_decider(game, max_depth)
    _, best_move = alphabeta_decider(game, max_depth)
    # _, best_move = mtd_f(game, 0, max_depth)
    return best_move

def minmax_decider(
    game, 
    max_depth, 
    maximizing_player=True
):
    """
    MinMax Decider algorithm for selecting the best move for the AI player.

    Parameters:
        game (OthelloGame): The current game state.当前游戏局势
        max_depth (int): The maximum search depth for the Alpha-Beta algorithm.
        Alpha-Beta剪枝的最大深度
        maximizing_player (bool): True if maximizing player (AI), False if minimizing player (opponent).
        用这个参数来控制当前是要选最大值还是最小值，从而把两个（min和max）函数合在一起。
    Returns:
        tuple: A tuple containing the evaluation value of the best move and the corresponding move (row, col).
        最好步骤的评估值和对应的走法。
    """
    if max_depth == 0 or game.is_game_over(): #如果已经游戏结束或者达到了最大的深度限制：
        return evaluate_game_state(game), None

    valid_moves = game.get_valid_moves()

    if maximizing_player: #如果是AI走，需要获取最大值
        max_eval = float("-inf")
        best_move = None

        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player #以上三行为创建一个当前游戏状态的副本
            new_game.make_move(*move) #move包括行坐标和列坐标，走了一步

            eval, _ = minmax_decider(new_game, max_depth - 1, False) #递归

            if eval > max_eval:
                max_eval = eval
                best_move = move

        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = None

        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            eval, _ = minmax_decider(new_game, max_depth - 1, True)

            if eval < min_eval:
                min_eval = eval
                best_move = move

        return min_eval, best_move

def alphabeta_decider(
    game, max_depth, maximizing_player=True, alpha=float("-inf"), beta=float("inf")
):
    """
    MinMax Decider algorithm for selecting the best move for the AI player.
    """
    # Your implementation for Alpha beta pruning
    if max_depth ==0 or game.is_game_over():
        return evaluate_game_state(game), None
    valid_moves = game.get_valid_moves()
    if maximizing_player: #如果是AI走，需要获取最大值
        max_eval = float("-inf")
        best_move = None
        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)
            eval, _ = alphabeta_decider(new_game, max_depth - 1, False,alpha,beta) #递归
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha= max (alpha,eval)
            if alpha >= beta:
                break
        return max_eval, best_move

    else:
        min_eval = float("inf")
        best_move = None
        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)
            eval, _ = alphabeta_decider(new_game, max_depth - 1, True,alpha,beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta=min(beta,eval)
            if alpha>=beta:
                break
        return min_eval, best_move

def mtd_f(game, guess, max_depth):
    """
    MTD(f) algorithm for selecting the best move for the AI player.
    
    Parameters:
        game (OthelloGame): The current game state.
        guess (float): The initial guess for the evaluation value.
        max_depth (int): The maximum search depth for the Alpha-Beta algorithm.

    Returns:
        tuple: A tuple containing the evaluation value of the best move and the corresponding move (row, col).
    """
    # Initialize alpha and beta bounds
    lower_bound = float("-inf")
    upper_bound = float("inf")
    
    g = guess
    
    while lower_bound < upper_bound:
        if g == lower_bound:
            beta = g + 1
        else:
            beta = g
        
        # Perform a zero-window search using alpha-beta pruning
        g, best_move = alphabeta_decider(game, max_depth, alpha=beta - 1, beta=beta, maximizing_player=True)
        
        # Update the bounds based on the result
        if g < beta:
            upper_bound = g
        else:
            lower_bound = g
    
    return g, best_move

def get_position_weights(game):
    total_disks = sum(1 for row in game.board for cell in row if cell != 0)
    game_phase = total_disks / 64  # 0-1, 0=开局, 1=终局

    if game_phase < 0.3:  # 开局
        return  [[500,-25,10,5 ,5 ,10,-25,500],
                [-25,-45,1 ,1 ,1 ,1 ,-45,-25],
                [10 ,1  ,3 ,2 ,2 ,3 ,1  ,10 ],
                [5  ,1  ,2 ,1 ,1 ,2 ,1  ,5  ],
                [5  ,1  ,2 ,1 ,1 ,2 ,1  ,5  ],
                [10 ,1  ,3 ,2 ,2 ,3 ,1  ,10 ],
                [-25,-45,1 ,1 ,1 ,1,-45 ,-25],
                [500,-25,10,5 ,5 ,10,-25,500]]
    elif game_phase < 0.7:
        return [[500, -25, 10, 5, 5, 10, -25, 500],
                [-25, -45, 1, 1, 1, 1, -45, -25],
                [10, 1, 3, 2, 2, 3, 1, 10],
                [5, 1, 2, 1, 1, 2, 1, 5],
                [5, 1, 2, 1, 1, 2, 1, 5],
                [10, 1, 3, 2, 2, 3, 1, 10],
                [-25, -45, 1, 1, 1, 1, -45, -25],
                [500, -25, 10, 5, 5, 10, -25, 500]]

def evaluate_game_state(game):
    """
    Evaluates the current game state for the AI player.
    Parameters:
        game (OthelloGame): The current game state.
    Returns:
        float: The evaluation value representing the desirability of the game state for the AI player.
    """
    weights=get_dynamic_weights(game)
    POSITION_WEIGHTS = get_position_weights(game)

    # Coin parity (difference in disk count)
    player_disk_count = sum(row.count(game.current_player) for row in game.board)
    opponent_disk_count = sum(row.count(-game.current_player) for row in game.board)
    coin_parity = player_disk_count - opponent_disk_count

    # Mobility (number of valid moves for the current player)
    player_valid_moves = len(game.get_valid_moves())

    temp_game = copy.deepcopy(game)
    temp_game.current_player = -game.current_player
    opponent_valid_moves = len(temp_game.get_valid_moves())

    mobility = player_valid_moves - opponent_valid_moves

    # Stability (number of stable disks)
    stability = calculate_stability(game)

    # Positional_score
    positional_score = 0
    for i in range(8):
        for j in range(8):
            if game.board[i][j]==game.current_player:
                positional_score+=POSITION_WEIGHTS[i][j]
            elif game.board[i][j]==game.current_player:
                positional_score -= POSITION_WEIGHTS[i][j]

    # Combine the factors with the corresponding weights to get the final evaluation value
    evaluation = (
        coin_parity * weights['coin_parity']#_weight
        + mobility * weights['mobility']#_weight
        + stability * weights['stability']#_weight
        + positional_score * weights['positional_score']#_weight
    )

    return evaluation

def get_dynamic_weights(game):
    """根据游戏阶段动态调整权重"""
    total_disks = sum(1 for row in game.board for cell in row if cell != 0)
    game_phase = total_disks / 64  # 0-1, 0=开局, 1=终局

    if game_phase < 0.3:  # 开局
        return {
            'coin_parity': 0.5,  # 开局棋子数量不重要
            'mobility': 3.0,  # 行动力很重要
            'stability': 1.0,  # 稳定性不太重要
            'positional_score':5.0
        }
    elif game_phase < 0.7:  # 中局
        return {
            'coin_parity': 1.0,
            'mobility': 2.0,
            'stability': 3.0,
            'positional_score': 1.0
        }
    else:  # 终局
        return {
            'coin_parity': 3.0,  # 棋子数量最重要
            'mobility': 0.5,  # 行动力不重要
            'stability': 2.0,
            'positional_score': 1.0
        }

def calculate_stability(game):
    """
    Calculates the stability of the AI player's disks on the board.
    Parameters:
        game (OthelloGame): The current game state.
    Returns:
        int: The number of stable disks for the AI player.
    """
    BOARD_SIZE=8

    # 使用集合记录已确认的稳定子
    confirmed_stable = set()

    def is_valid_position(row, col):
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    # 初始稳定子：角落棋子
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for r, c in corners:
        if game.board[r][c] == game.current_player:
            confirmed_stable.add((r, c))

    def is_stable_in_direction(row, col, dr, dc):
        """
        判断棋子在某个方向上是否稳定
        """
        # 检查相邻位置
        r, c = row + dr, col + dc

        # 情况1：紧邻边界 → 稳定
        if not is_valid_position(r, c):
            return True

        # 情况2：紧邻已确认的稳定子 → 稳定
        if (r, c) in confirmed_stable:
            return True

        # 如果相邻位置是空位或对手棋子，这个方向不稳定
        if game.board[r][c] != game.current_player:
            return False

        # 递归检查相邻的同色棋子是否稳定
        return is_stable_in_direction(r, c, dr, dc)

    def is_fully_stable(row, col):
        """判断棋子是否在四个方向都稳定"""
        directions =  [
            (0, 1), (1, 0), (0, -1), (-1, 0),   # 正交方向
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线方向
        ]

        # 计算稳定方向的数量
        stable_directions = 0
        for dr, dc in directions:
            if is_stable_in_direction(row, col, dr, dc):
                stable_directions += 1

        # 放宽条件：多数方向稳定即可认为是稳定子
        return stable_directions >= 6  # 可以根据需要调整阈值

    # 稳定性传播：从已确认的稳定子开始，寻找新的稳定子
    changed = True
    while changed:
        changed = False
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r, c) in confirmed_stable:
                    continue  # 已经是稳定子，跳过
                if game.board[r][c] == game.current_player:
                    if is_fully_stable(r, c):
                        confirmed_stable.add((r, c))
                        changed = True

    return len(confirmed_stable)