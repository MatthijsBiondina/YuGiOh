from tkinter import Frame, Tk

import cv2
import pygame

from src.utils.tools import pyout


class GUI:
    BACKGROUND_COLOR = (45, 53, 69)
    TEXT_COLOR = (209, 191, 152)

    def __init__(self):
        pygame.init()
        logo = pygame.image.load("res/icon.png")
        pygame.display.set_icon(logo)
        pygame.display.set_caption("CardMarket")
        pygame.font.init()

        self.screen = pygame.display.set_mode((1980, 1060), pygame.RESIZABLE)
        self.font = pygame.font.SysFont('Ubuntu Condensed', 30)

        self.running = True
        self.total_price = 0.

        self.camera_img = None
        self.process_img = None
        self.card_name = None
        self.card_set = None
        self.card_quality = None
        self.card_price = None
        self.quality = None
        self.all_card_sets = None

    def update(self):
        if not self.running:
            return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.running = False
                return False

        self.screen.fill(self.BACKGROUND_COLOR)
        self.screen.blit(self.camera_img, (50, 50))

        text_surf = self.font.render(f"$ {self.total_price:.2f}", True, self.TEXT_COLOR)
        self.screen.blit(text_surf, (50, 664))

        if self.process_img is not None:
            self.screen.blit(self.process_img, (521, 50))
        if self.card_name is not None:
            text_surf = self.font.render(self.card_name, True, self.TEXT_COLOR)
            self.screen.blit(text_surf, (992, 50))
        if self.card_set is not None:
            text_surf = self.font.render(self.card_set, True, self.TEXT_COLOR)
            self.screen.blit(text_surf, (992, 100))
        if self.card_price is not None:
            text_surf = self.font.render(f"$ {self.card_price:.2f}", True, self.TEXT_COLOR)
            self.screen.blit(text_surf, (992, 150))
        if self.quality is not None:
            text_surf = self.font.render(self.quality, True, self.TEXT_COLOR)
            self.screen.blit(text_surf, (992, 200))

        pygame.display.update()
        return True

    def set_camera_img(self, img):
        img = cv2.resize(img, (421, 614)).transpose(1, 0, 2)[:, :, ::-1]
        self.camera_img = pygame.surfarray.make_surface(img)

    def set_process_img(self, img):
        if img is None:
            self.process_img = None
        else:
            img = cv2.resize(img, (421, 614)).transpose(1, 0, 2)[:, :, ::-1]
            self.process_img = pygame.surfarray.make_surface(img)

    def set_card_name(self, name):
        self.card_name = name

    def set_card_set(self, cardset):
        if cardset is None:
            self.card_set = None
        else:
            self.all_card_sets = cardset
            self.card_set = cardset[0]['set_code']

    def query_quality(self, card):
        self.quality = 'Good'
        orig_price = self.card_price
        self.card_price = 0.8 * orig_price
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.running = False
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_KP_0:
                        self.quality = 'Poor'
                        self.card_price = 0.
                    if event.key == pygame.K_KP_1:
                        self.quality = 'Played'
                        self.card_price = 0.6 * orig_price
                    if event.key == pygame.K_KP_2:
                        self.quality = 'Light Played'
                        self.card_price = 0.7 * orig_price
                    if event.key == pygame.K_KP_3:
                        self.quality = 'Good'
                        self.card_price = 0.8 * orig_price
                    if event.key == pygame.K_KP_4:
                        self.quality = 'Excellent'
                        self.card_price = 0.9 * orig_price
                    if event.key == pygame.K_KP_5:
                        self.quality = 'Near Mint'
                        self.card_price = 1. * orig_price
                    if event.key == pygame.K_KP_ENTER:
                        quality = self.quality
                        card_price = self.card_price
                        self.quality = None
                        self.card_price = None
                        return quality, card_price
                    if event.key == pygame.K_BACKSPACE:
                        self.quality = None
                        self.card_price = None
                        return False
                    if event.key == pygame.K_SPACE:
                        return self.dropdown_cardset(card)



            self.update()


    def dropdown_cardset(self, card):
        import tkinter as tk
        from tkinter import N, W, E, S, StringVar

        value = self.card_set


        def select():

            root.destroy()

        root = tk.Tk()
        # use width x height + x_offset + y_offset (no spaces!)
        root.geometry("%dx%d+%d+%d" % (330, 80, 200, 150))
        root.title("Manually select set")

        var = tk.StringVar(root)
        # initial value
        var.set(self.card_set)


        choices = sorted([s['set_code'] for s in card.card_sets])
        option = tk.OptionMenu(root, var, *choices)
        option.pack(side='left', padx=10, pady=10)

        button = tk.Button(root, text="check value slected", command=select)
        button.pack(side='left', padx=20, pady=10)

        root.mainloop()

        return var.get()

    def set_price(self, price):
        self.card_price = price

    def set_total_price(self, price):
        self.total_price = price
