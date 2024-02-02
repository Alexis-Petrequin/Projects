/*
** EPITECH PROJECT, 2020
** free2
** File description:
** about free
*/

#include "../../include/mylist.h"

static void free_map3(map_t *map)
{
    sfText_destroy(map->presse);
    sfSound_destroy(map->sound);
    sfSoundBuffer_destroy(map->s_buffer);
    for (int i = 0; map->scenary[i] != NULL; i++) {
        map->scenary[i]->sprite;
        free(map->scenary[i]);
    }
    free(map->scenary);
    for (int i = 0; map->texture[i] != NULL; i++)
        sfTexture_destroy(map->texture[i]);
    free(map->texture);
    free(map);
}

void free_map2(map_t *map)
{
    for (int i = 0; map->menu->rectangle[i] != NULL; i++)
        sfRectangleShape_destroy(map->menu->rectangle[i]);
    free(map->menu->rectangle);
    for (int i = 0; map->menu->text[i] != NULL; i++)
        sfText_destroy(map->menu->text[i]);
    free(map->menu->text);
    free(map->menu);
    sfMusic_destroy(map->music);
    sfSprite_destroy(map->perso->map);
    sfSprite_destroy(map->perso->perso);
    free(map->perso);
    for (int i = 0; map->pnj[i] != NULL; i++) {
        sfSprite_destroy(map->pnj[i]->sprite);
        free(map->pnj[i]);
    }
    free(map->pnj);
    free_map3(map);
}