/*
** EPITECH PROJECT, 2020
** battle_init
** File description:
** about init battle
*/

#include "../../include/mylist.h"

sprite_t **init_sprite_battle(sfTexture **text)
{
    sprite_t **sprite = malloc(sizeof(sprite_t *) * 12);
    int nb[11] = { 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    sfvector4f_t posize[11] = { { 0, 0, 3, 3 }, { 0, -465, 2.07, 1.8 },
    { 0, -465, 2.07, 1.8 }, { 0, -465, 2.07, 1.8 }, { 0, -465, 2.07, 1.8 },
    { 0, -465, 2.07, 1.8 }, { 0, -465, 2.07, 1.8 }, { 0, -465, 2.07, 1.8 },
    { 0, -280, 2.07, 1.4 }, { 0, -835, 2.07, 2.3 }, { 0, -360, 2.07, 1.8 } };
    sfVector2f sp[11] = { { 0, 0 }, { -0.5, 0 }, { -0.5, 0 }, { -1, 0 },
    { -1, 0 }, { -2, 0 }, { -2, 0 }, { -3, 0 }, { -3, 0 }, { 0, 0 },
    { 0, 0 } };

    if (!sprite)
        return (NULL);
    for (int i = 0; i <= 10; i++)
        if (!(sprite[i] = make_sprite(text[nb[i]], posize[i], sp[i])))
            return (NULL);
    sprite[11] = NULL;
    return (sprite);
}

button_t *button_battle(char *text, sfvector4f_t po, ressource_t *re, int si)
{
    button_t *button = malloc(sizeof(button_t));
    sfVector2f pos = { po.x, po.y };
    sfVector2f size = { po.a, po.b };
    sfVector2f origin = { po.a / 2, po.b / 2 };
    sfVector2f origin2 = { 0, 90 * ( si * 0.01 - 0.05) };

    button->sprite = sfRectangleShape_create();
    sfRectangleShape_setTexture(button->sprite, re->texture[0], sfFalse);
    sfRectangleShape_setSize(button->sprite, size);
    sfRectangleShape_setOrigin(button->sprite, origin);
    sfRectangleShape_setPosition(button->sprite, pos);
    button->text = sfText_create();
    sfText_setFont(button->text, re->police);
    sfText_setString(button->text, text);
    sfText_setCharacterSize(button->text, si);
    sfText_setColor(button->text, sfYellow);
    origin2.x = sfText_getLocalBounds(button->text).width / 2;
    sfText_setOrigin(button->text, origin2);
    sfText_setPosition(button->text, pos);
    return (button);
}

void butt2(ressource_t *ress, button_t **button, sfvector4f_t *posize)
{
    char str[27][30] = { {""}, {"attack"}, {"defend"}, {"items"}, {"leave"},
    {"cut"}, {"fire"}, {"heal"}, {"back"}, {"hp"}, {"mana"}, {"fire"},
    {"back"}, {"100"}, {"100"}, {"100"}, {"100"}, {"000"}, {"000"},
    {"GAME OVER"}, {"EXIT"}, {"YOU WIN !"}, {"You get a mana potion"},
    {"Continue"}, {"Go to main menu ?"}, {"Yes"}, {"No"} };

    for (int i = 0; i <= 12; i++)
        button[i] = button_battle(str[i], posize[i], ress, 55);
    for (int i = 13; i <= 16; i++)
        button[i] = button_battle(str[i], posize[i], ress, 20);
    for (int i = 17; i <= 18; i++)
        button[i] = button_battle(str[i], posize[i], ress, 30);
    for (int i = 20; i <= 26; i++)
        if (i == 20 || i == 22 || i == 23 || i == 25 || i == 26)
            button[i] = button_battle(str[i], posize[i], ress, 50);
    button[19] = button_battle(str[19], posize[19], ress, 80);
    button[21] = button_battle(str[21], posize[21], ress, 120);
    button[24] = button_battle(str[24], posize[24], ress, 70);
    button[27] = NULL;
}

button_t **init_battle_butt(ressource_t *ress)
{
    button_t **button = malloc(sizeof(button_t *) * 28);
    sfvector4f_t posize[27] = { {1590, 910, 500, 200}, {1470, 865, 200, 80},
    {1710, 865, 200, 80}, {1470, 960, 200, 80}, {1710, 960, 200, 80},
    {3470, 865, 200, 80}, {3710, 865, 200, 80}, {3470, 960, 200, 80},
    {3710, 960, 200, 80}, {-530, 865, 200, 80}, {-290, 865, 200, 80},
    {-530, 960, 200, 80}, {-290, 960, 200, 80}, {320, 930, 150, 20},
    {320, 950, 150, 20}, {980, 930, 120, 20}, {980, 950, 120, 20},
    {450, 640, 2, 2}, {860, 670, 2, 2}, {-1000, 0, 500, 200},
    {-1000, 0, 200, 80}, {-1000, 0, 600, 200}, {-1000, 0, 600, 80},
    {-1000, 0, 230, 90}, {-1000, 0, 600, 200}, {-1000, 0, 230, 90},
    {-1000, -1000, 230, 90}};

    butt2(ress, button, posize);
    sfRectangleShape_setFillColor(button[16]->sprite, sfBlue);
    sfRectangleShape_setFillColor(button[14]->sprite, sfBlue);
    sfRectangleShape_setFillColor(button[15]->sprite, sfRed);
    sfRectangleShape_setFillColor(button[13]->sprite, sfRed);
    sfRectangleShape_setFillColor(button[17]->sprite, sfTransparent);
    sfRectangleShape_setFillColor(button[18]->sprite, sfTransparent);
    return (button);
}

battle_t *init_battle(void)
{
    battle_t *menu = malloc(sizeof(battle_t));

    menu->ress = init_ress_battle();
    menu->sprite = init_sprite_battle(menu->ress->texture);
    menu->event = malloc(sizeof(sfEvent));
    menu->clock = init_clock(5);
    menu->mob = init_mob(menu->ress->texture);
    menu->button = init_battle_butt(menu->ress);
    menu->sound = sfSound_create();
    menu->fire = 0;
    sfSound_setBuffer(menu->sound, menu->ress->s_buffer);
    return (menu);
}