__author__ = 'liuzhen'



import os

import pygame



chinese_dir = 'D:\\kwai_paper\\first\\chinese'

if not os.path.exists(chinese_dir):

    os.mkdir(chinese_dir)



pygame.init()

start, end = (0x4E00, 0x9FA5)

for codepoint in range(int(start), int(end)):

    word = unichr(codepoint)

    font = pygame.font.Font("yahei.ttf", 70)

    rtext = font.render(word, True, (255, 255, 255), (0, 0, 0))

    pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))