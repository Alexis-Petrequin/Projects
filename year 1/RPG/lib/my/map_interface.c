/*
** EPITECH PROJECT, 2019
** map_interface
** File description:
** about interface of map
*/

#include "../../include/mylist.h"

static void init_interface2(button_t **in)
{
    sfVector2f pos = { -1000, -1000 };

    for (int i = 0; i < 7; ++i) {
        sfRectangleShape_setOrigin(in[i]->sprite,
        (sfVector2f){ sfRectangleShape_getGlobalBounds
        (in[i]->sprite).width / 2,
        sfRectangleShape_getGlobalBounds(in[i]->sprite).height / 2 });
        sfRectangleShape_setPosition(in[i]->sprite, pos);
        sfText_setOrigin(in[i]->text,
        (sfVector2f){ sfText_getGlobalBounds(in[i]->text).width / 2, 40 });
        sfText_setPosition(in[i]->text, pos);
    }
}

button_t **init_interface(map_t *map)
{
    button_t **interface = malloc(sizeof(button_t *) * 8);
    char text[7][25] = { {"Status"}, {"inventory"}, {"main menu"},
    {"yes"}, {"no"}, {"Game Over"}, {"Finish"} };

    if (!interface)
        return (NULL);
    for (int i = 0; i < 7; ++i) {
        interface[i] = malloc(sizeof(button_t));
        interface[i]->sprite = sfRectangleShape_create();
        interface[i]->text = sfText_create();
        sfRectangleShape_setTexture(interface[i]->sprite,
        map->texture[10], sfTrue);
        sfRectangleShape_setSize(interface[i]->sprite, (sfVector2f){250, 80});
        sfText_setFont(interface[i]->text, map->font);
        sfText_setString(interface[i]->text, text[i]);
        sfText_setCharacterSize(interface[i]->text, 55);
        sfText_setColor(interface[i]->text, sfYellow);
    }
    init_interface2(interface);
    interface[7] = NULL;
    return (interface);
}