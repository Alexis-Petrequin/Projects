/*
** EPITECH PROJECT, 2020
** menu_init
** File description:
** about init menu
*/

#include "../../include/mylist.h"

button_t *button_menu(char *text, sfvector4f_t posize, ressource_t *ress)
{
    button_t *button = malloc(sizeof(button_t));
    sfVector2f pos = { posize.x, posize.y };
    sfVector2f size = { posize.a, posize.b };
    sfVector2f origin = {posize.a / 2, posize.b / 2};
    sfVector2f origin2 = { 0, 90 * 0.7 };

    button->sprite = sfRectangleShape_create();
    sfRectangleShape_setTexture(button->sprite, ress->texture[0], sfFalse);
    sfRectangleShape_setSize(button->sprite, size);
    sfRectangleShape_setOrigin(button->sprite, origin);
    sfRectangleShape_setPosition(button->sprite, pos);
    button->text = sfText_create();
    sfText_setFont(button->text, ress->police);
    sfText_setString(button->text, text);
    sfText_setCharacterSize(button->text, 90);
    sfText_setColor(button->text, sfYellow);
    origin2.x = sfText_getLocalBounds(button->text).width / 2;
    sfText_setOrigin(button->text, origin2);
    sfText_setPosition(button->text, pos);
    return (button);
}

button_t **init_menu_butt(ressource_t *ress)
{
    button_t **button = malloc(sizeof(button_t *) * 17);
    sfvector4f_t posize[17] = { {950, 500, 500, 150}, {950, 700, 500, 150},
    {950, 900, 500, 150}, {-2050, 500, 500, 150}, {-1700, 500, 80, 90},
    {-1300, 500, 80, 90}, {-2050, 700, 500, 150}, {-1500, 500, 230, 115},
    {-1700, 700, 80, 100}, {-1300, 700, 80, 100}, {-1500, 700, 230, 115},
    {-2050, 900, 500, 150}, {3650, 500, 500, 150}, {4300, 500, 500, 150},
    {3950, 900, 500, 150}, {3950, 700, 600, 150}, { 0 } };
    char *str[17] = { "PLAY", "SETTINGS", "LEAVE", "MUSIC", "-",
    "+", "SOUND", "70", "-", "+", "100", "BACK",
    "NEW GAME", "CONTINUE", "BACK", "HOW TO PLAY", NULL };

    if (button == NULL || posize == NULL || str == NULL)
        return (NULL);
    for (int i = 0; i <= 15; i++)
        button[i] = button_menu(str[i], posize[i], ress);
    button[16] = NULL;
    return (button);
}

sprite_t **init_sprite_menu(sfTexture **texture)
{
    sprite_t **sprite = malloc(sizeof(sprite_t *) * 10);
    int nb[9] = { 1, 2, 2, 3, 3, 4, 4, 5, 6 };
    sfvector4f_t posize[9] = { { 0, 0, 1, 1 }, { 0, 0, 1, 1 },
    { 1920, 0, 1, 1 }, { 0, 0, 1, 1 }, { 1920, 0, 1, 1 }, { 0, 0, 1, 1 },
    { 1920, 0, 1, 1 }, { 0, 0, 1, 1 }, { 0, 0, 1, 1 } };
    sfVector2f sp[9] = { { 0, 0 }, { -0.5, 0 }, { -0.5, 0 }, { -1, 0 },
    { -1, 0 }, { -2, 0 }, { -2, 0 }, { -3, 0 }, { -3, 0 } };

    if (!sprite)
        return (NULL);
    for (int i = 0; i <= 8; i++)
        if (!(sprite[i] = make_sprite(texture[nb[i]], posize[i], sp[i])));
    sprite[9] = NULL;
    return (sprite);
}

menu_t *init_menu(void)
{
    menu_t *menu = malloc(sizeof(menu_t));

    if (!menu)
        return (NULL);
    if ((menu->ress = init_ress_menu()) == NULL)
        return (NULL);
    menu->button = init_menu_butt(menu->ress);
    menu->event = malloc(sizeof(sfEvent));
    menu->sound = sfSound_create();
    menu->sprite = init_sprite_menu(menu->ress->texture);
    menu->clock = sfClock_create();
    if (!menu->button || !menu->event || !menu->sound ||
    !menu->sprite || !menu->clock)
        return (NULL);
    sfSound_setBuffer(menu->sound, menu->ress->s_buffer);
    sfMusic_setLoop(menu->ress->music, sfTrue);
    sfMusic_setVolume(menu->ress->music, 100);
    sfSound_setVolume(menu->sound, 100);
    return (menu);
}