import os
import sys
import pygame
import pickle
import argparse
from play import AliensEnvPygame
from learn import extract_features

def main():
    pygame.init()
    env = AliensEnvPygame(level=0, render=False)

    #解析参数
    parser = argparse.ArgumentParser(description="Aliens 游戏，请选择测试模型")
    parser.add_argument(
        "--model",
        choices=["rf", "svc", "mlp", "xg"],
        required=True,
        help="测试模型:rf--随机森林；svc--支持向量机；mlp--多层感知器；xg--梯度提升。"
    )
    args = parser.parse_args()

    # 加载模型
    model_path = f'models/{args.model}_lvl0/gameplay_model.pkl' # 替换为你的模型的路径
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    print("模型加载完成")

    observation = env.reset()

    grid_image = env.do_render()

    mode = grid_image.mode
    size = grid_image.size
    data_image = grid_image.tobytes()
    pygame_image = pygame.image.fromstring(data_image, size, mode)

    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Aliens Game - AI Playing')

    screen.blit(pygame_image, (0, 0))
    pygame.display.flip()

    done = False
    total_score = 0
    step = 0
    while not done:
        features = extract_features(observation)
        features = features.reshape(1, -1)

        action = clf.predict(features)[0]
        if args.model=='xg':
            action+=1

        observation, reward, game_over, info = env.step(action)
        total_score += reward
        print(f"Step: {step}, Action taken: {action}, Reward: {reward}, Done: {game_over}, Info: {info}")
        step += 1

        grid_image = env.do_render()
        mode = grid_image.mode
        size = grid_image.size
        data_image = grid_image.tobytes()
        pygame_image = pygame.image.fromstring(data_image, size, mode)

        screen.blit(pygame_image, (0, 0))
        pygame.display.flip()

        if game_over or step > 500:
            print("游戏结束!")
            print(f"信息: {info}，分数：{total_score}")
            done = True

        pygame.time.delay(100)
    os.makedirs(f'logs/{args.model}_records_lvl{env.level}_{env.timing}',exist_ok=True)
    env.save_gif(path= f'logs/{args.model}_records_lvl{env.level}_{env.timing}',filename=f'replay_ai.gif')

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
