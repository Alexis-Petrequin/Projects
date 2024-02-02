/*
** EPITECH PROJECT, 2020
** free
** File description:
** about free
*/

#include "../../include/mylist.h"

static void free_menu(menu_t *menu)
{
    free(menu->event);
    for (int i = 0; menu->button[i] != NULL; i++) {
        sfRectangleShape_destroy(menu->button[i]->sprite);
        sfText_destroy(menu->button[i]->text);
        free(menu->button[i]);
    }
    free(menu->button);
    for (int i = 0; menu->sprite[i] != NULL; i++) {
        sfSprite_destroy(menu->sprite[i]->sprite);
        free(menu->sprite[i]);
    }
    free(menu->sprite);
    sfSound_destroy(menu->sound);
    for (int i = 0; menu->ress->texture[i] != NULL; i++)
        sfTexture_destroy(menu->ress->texture[i]);
    sfMusic_destroy(menu->ress->music);
    sfFont_destroy(menu->ress->police);
    sfSoundBuffer_destroy(menu->ress->s_buffer);
    sfClock_destroy(menu->clock);
    free(menu->ress);
    free(menu);
}

static void free_battle2(battle_t *battle)
{
    free(battle->clock);
    free(battle->event);
    for (int i = 0; battle->mob[i] != NULL; i++) {
        sfSprite_destroy(battle->mob[i]->sprite);
        free(battle->mob[i]);
    }
    free(battle->mob);
    for (int i = 0; battle->sprite[i] != NULL; i++) {
        sfSprite_destroy(battle->sprite[i]->sprite);
        free(battle->sprite[i]);
    }
    free(battle->sprite);
    sfSound_destroy(battle->sound);
    sfFont_destroy(battle->ress->police);
    for (int i = 0; battle->ress->texture[i] != NULL; i++) {
        sfTexture_destroy(battle->ress->texture[i]);
    }
    free(battle->ress->texture);
    sfMusic_destroy(battle->ress->music);
    sfSoundBuffer_destroy(battle->ress->s_buffer);
}

static void free_battle(battle_t *battle)
{
    for (int i = 0; battle->button[i] != NULL; i++) {
        sfRectangleShape_destroy(battle->button[i]->sprite);
        sfText_destroy(battle->button[i]->text);
        free(battle->button[i]);
    }
    free(battle->button);
    for (int i = 0; battle->clock[i] != NULL; i++)
        sfClock_destroy(battle->clock[i]);
    free_battle2(battle);
    free(battle);
}

static void free_map(map_t *map)
{
    for (int i = 0; map->bubble[i] != NULL; i++) {
        sfRectangleShape_destroy(map->bubble[i]->sprite);
        sfText_destroy(map->bubble[i]->text);
        free(map->bubble[i]);
    }
    free(map->bubble);
    sfView_destroy(map->camera);
    for (int i = 0; map->clock[i] != NULL; i++)
        sfClock_destroy(map->clock[i]);
    free(map->clock);
    free(map->event);
    sfFont_destroy(map->font);
    for (int i = 0; map->interface[i] != NULL; i++) {
        sfRectangleShape_destroy(map->interface[i]->sprite);
        sfText_destroy(map->interface[i]->text);
        free(map->interface[i]);
    }
    free(map->interface);
    free_map2(map);
}

void free_all(player_t *player, menu_t *menu, battle_t *battle, map_t *map)
{
    free_menu(menu);
    free_battle(battle);
    free_map(map);
    free(player);
}