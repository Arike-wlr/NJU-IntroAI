from othello_game import OthelloGame
import copy

def get_best_move(game, max_depth=8):
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
    #_, best_move = alphabeta_decider(game, max_depth)
    _, best_move = mtd_f(game, 0, max_depth)
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
    """
    统一数量级的位置权重表
    数值范围：[-0.3, 0.6]，确保各维度平衡
    """
    total_disks = sum(1 for row in game.board for cell in row if cell != 0)
    game_phase = total_disks / 64  # 0-1, 0=开局, 1=终局

    if game_phase < 0.3:  # 开局阶段
        return [
            # 开局：强调角落，避免C位，控制边界
            [0.6  , -0.25, 0.15 , 0.08 , 0.08 , 0.15 , -0.25, 0.6  ],
            [-0.25, -0.3 , -0.08, -0.08, -0.08, -0.08, -0.3 , -0.25],
            [0.15 , -0.08, 0.05 , 0.02 , 0.02 , 0.05 , -0.08, 0.15 ],
            [0.08 , -0.08, 0.02 , 0.01 , 0.01 , 0.02 , -0.08, 0.08 ],
            [0.08 , -0.08, 0.02 , 0.01 , 0.01 , 0.02 , -0.08, 0.08 ],
            [0.15 , -0.08, 0.05 , 0.02 , 0.02 , 0.05 , -0.08, 0.15 ],
            [-0.25, -0.3 , -0.08, -0.08, -0.08, -0.08, -0.3 , -0.25],
            [0.6  , -0.25, 0.15 , 0.08 , 0.08 , 0.15 , -0.25, 0.6  ]
        ]

    elif game_phase < 0.7:  # 中局阶段
        return [
            # 中局：平衡发展，内部位置价值提升
            [0.5, -0.15, 0.2, 0.12, 0.12, 0.2, -0.15, 0.5],  # 0
            [-0.15, -0.2, 0.03, 0.02, 0.02, 0.03, -0.2, -0.15],  # 1
            [0.2, 0.03, 0.1, 0.06, 0.06, 0.1, 0.03, 0.2],  # 2
            [0.12, 0.02, 0.06, 0.04, 0.04, 0.06, 0.02, 0.12],  # 3
            [0.12, 0.02, 0.06, 0.04, 0.04, 0.06, 0.02, 0.12],  # 4
            [0.2, 0.03, 0.1, 0.06, 0.06, 0.1, 0.03, 0.2],  # 5
            [-0.15, -0.2, 0.03, 0.02, 0.02, 0.03, -0.2, -0.15],  # 6
            [0.5, -0.15, 0.2, 0.12, 0.12, 0.2, -0.15, 0.5]  # 7
        ]

    else:  # 终局阶段
        return [
            # 终局：所有位置都有正价值，强调棋子数量
            [0.4, 0.08, 0.25, 0.15, 0.15, 0.25, 0.08, 0.4],  # 0
            [0.08, 0.05, 0.12, 0.09, 0.09, 0.12, 0.05, 0.08],  # 1
            [0.25, 0.12, 0.18, 0.14, 0.14, 0.18, 0.12, 0.25],  # 2
            [0.15, 0.09, 0.14, 0.11, 0.11, 0.14, 0.09, 0.15],  # 3
            [0.15, 0.09, 0.14, 0.11, 0.11, 0.14, 0.09, 0.15],  # 4
            [0.25, 0.12, 0.18, 0.14, 0.14, 0.18, 0.12, 0.25],  # 5
            [0.08, 0.05, 0.12, 0.09, 0.09, 0.12, 0.05, 0.08],  # 6
            [0.4, 0.08, 0.25, 0.15, 0.15, 0.25, 0.08, 0.4]  # 7
        ]


def get_dynamic_weights(game):
    """根据游戏阶段动态调整权重"""
    total_disks = sum(1 for row in game.board for cell in row if cell != 0)
    game_phase = total_disks / 64  # 0-1, 0=开局, 1=终局

    if game_phase < 0.3:  # 开局阶段 (0-19子)
        return {
            'coin_parity': 0.1,  # 棋子数量：不重要
            'mobility': 0.5,  # 行动力：最重要
            'stability': 0.2,  # 稳定性：次要
            'positional_score': 0.2  # 位置分数：重要
        }
    elif game_phase < 0.7:  # 中局阶段 (20-44子)
        return {
            'coin_parity': 0.25,  # 棋子数量：重要性提升
            'mobility': 0.3,  # 行动力：仍然重要
            'stability': 0.3,  # 稳定性：重要性提升
            'positional_score': 0.15  # 位置分数：重要性降低
        }
    else:  # 终局阶段 (45-64子)
        return {
            'coin_parity': 0.5,  # 棋子数量：最重要
            'mobility': 0.1,  # 行动力：不重要
            'stability': 0.25,  # 稳定性：重要
            'positional_score': 0.15  # 位置分数：辅助作用
        }

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
    coin_parity = 100*(player_disk_count - opponent_disk_count)/(player_disk_count + opponent_disk_count)

    # Mobility (number of valid moves for the current player)
    player_valid_moves = len(game.get_valid_moves())
    temp_game = copy.deepcopy(game)
    temp_game.current_player = -game.current_player
    opponent_valid_moves = len(temp_game.get_valid_moves())
    total_moves = player_valid_moves + opponent_valid_moves
    if total_moves == 0:
        mobility = 0
    else:
        mobility =100 *( player_valid_moves - opponent_valid_moves)/total_moves

    # Stability (number of stable disks)
    stability =1.5 * calculate_stability(game)

    # Positional_score
    positional_score = 0
    for i in range(8):
        for j in range(8):
            if game.board[i][j]==game.current_player:
                positional_score+=POSITION_WEIGHTS[i][j]
            elif game.board[i][j]==-game.current_player:
                positional_score -= POSITION_WEIGHTS[i][j]
    positional_score*=100

    # Combine the factors with the corresponding weights to get the final evaluation value
    evaluation = (
        coin_parity * weights['coin_parity']#_weight
        + mobility * weights['mobility']#_weight
        + stability * weights['stability']#_weight
        + positional_score * weights['positional_score']#_weight
    )

    return evaluation

def calculate_stability(game):
    """
    Calculates the stability of the AI player's disks on the board.
    Parameters:
        game (OthelloGame): The current game state.
    Returns:
        int: The number of stable disks for the AI player.
    """

    def first_stability(game):
        def neighbors(row, col):
            return [
                (row + dr, col + dc)
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if (dr, dc) != (0, 0) and 0 <= row + dr < 8 and 0 <= col + dc < 8
            ]

        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        edges = [(i, j) for i in [0, 7] for j in range(1, 7)] + [
            (i, j) for i in range(1, 7) for j in [0, 7]
        ]
        inner_region = [(i, j) for i in range(2, 6) for j in range(2, 6)]
        regions = [corners, edges, inner_region]
        # 定义棋盘区域。
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

    def second_stability(game):

        BOARD_SIZE = 8

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

            # 情况3：沿着方向检查是否完全被占据且没有空位
            current_r, current_c = r, c
            while is_valid_position(current_r, current_c):
                if game.board[current_r][current_c] == 0:  # 发现空位
                    return False
                current_r += dr
                current_c += dc

            # 情况4：走到边界都没有空位 → 稳定
            return True

        def is_fully_stable(row, col):
            """判断棋子是否在四个方向都稳定"""
            directions = [
                (0, 1), (1, 0), (0, -1), (-1, 0),  # 正交方向
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线方向
            ]

            # 计算稳定方向的数量
            stable_directions = 0
            for dr, dc in directions:
                if is_stable_in_direction(row, col, dr, dc):
                    stable_directions += 1

            # 放宽条件：多数方向稳定即可认为是稳定子
            return stable_directions >= 4  # 可以根据需要调整阈值

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

    total_disks = sum(1 for row in game.board for cell in row if cell != 0)
    game_phase = total_disks / 64  # 0-1, 0=开局, 1=终局

    if game_phase < 0.5:  # 开局
        return first_stability(game)
    else:
        return second_stability(game)