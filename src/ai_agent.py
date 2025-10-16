from othello_game import OthelloGame


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
            new_game.current_player = game.current_player #创建一个当前游戏状态的副本
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



def evaluate_game_state(game):
    """
    Evaluates the current game state for the AI player.
    Parameters:
        game (OthelloGame): The current game state.
    Returns:
        float: The evaluation value representing the desirability of the game state for the AI player.
    """
    weights=get_dynamic_weights(game)

    POSITION_WEIGHTS = [[500,-25,10,5 ,5 ,10,-25,500],
                        [-25,-45,1 ,1 ,1 ,1 ,-45,-25],
                        [10 ,1  ,3 ,2 ,2 ,3 ,1  ,10 ],
                        [5  ,1  ,2 ,1 ,1 ,2 ,1  ,5  ],
                        [5  ,1  ,2 ,1 ,1 ,2 ,1  ,5  ],
                        [10 ,1  ,3 ,2 ,2 ,3 ,1  ,10 ],
                        [-25,-45,1 ,1 ,1 ,1,-45 ,-25],
                        [500,-25,10,5 ,5 ,10,-25,500]]

    # Coin parity (difference in disk count)
    player_disk_count = sum(row.count(game.current_player) for row in game.board)
    opponent_disk_count = sum(row.count(-game.current_player) for row in game.board)
    coin_parity = player_disk_count - opponent_disk_count

    # Mobility (number of valid moves for the current player)
    player_valid_moves = len(game.get_valid_moves())
    opponent_valid_moves = len(
        OthelloGame(player_mode=-game.current_player).get_valid_moves()
    )
    mobility = player_valid_moves - opponent_valid_moves

    # Stability (number of stable disks)
    stability = calculate_stability(game)

    # Positional_score
    positional_score=0 #TODO:遍历棋盘

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

    def neighbors(row, col):
        return [
            (row + dr, col + dc)
            for dr in [-1, 0, 1]
            for dc in [-1, 0, 1]
            if (dr, dc) != (0, 0) and 0 <= row + dr < 8 and 0 <= col + dc < 8
        ]

    # 划分棋盘区域。
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    edges = [(i, j) for i in [0, 7] for j in range(2, 6)] + [
        (i, j) for i in range(2, 6) for j in [0, 7]
    ]
    inner_region = [(i, j) for i in range(2, 6) for j in range(2, 6)]
    regions = [corners, edges, inner_region]

    stable_count = 0

    def is_stable_disk(row, col):
        return (
            all(game.board[r][c] == game.current_player for r, c in neighbors(row, col))
            or (row, col) in edges + corners
        )

    for region in regions:
        for row, col in region:
            if game.board[row][col] == game.current_player and is_stable_disk(row, col):
                stable_count += 1

    return stable_count
