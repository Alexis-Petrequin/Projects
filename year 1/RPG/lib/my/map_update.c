/*
** EPITECH PROJECT, 2019
** map_update
** File description:
** about map_update
*/

#include "../../include/mylist.h"

int update_statut(map_t *map, player_t *pl)
{
    char str[5] = {'\0'};

    sfText_setString(map->menu->text[3], numtostr(85 + pl->level * 15, str));
    sfText_setString(map->menu->text[5], numtostr(45 + pl->level * 5, str));
    sfText_setString(map->menu->text[7], numtostr(10 + pl->level * 2, str));
    sfText_setString(map->menu->text[9], numtostr(2 + pl->level * 2, str));
    sfText_setString(map->menu->text[11], numtostr(1 + pl->level, str));
    sfText_setString(map->menu->text[13], numtostr(pl->level, str));
    sfText_setString(map->menu->text[16], numtostr(pl->monsterkilled, str));
    if (pl->quest == 1)
        return (17);
    else
        return (15);
}

void update_stuff(map_t *map, player_t *pl)
{
    char str[5] = {'\0'};

    if (pl->nbpothp > 5)
        pl->nbpothp = 5;
    if (pl->nbpotmana > 5)
        pl->nbpotmana = 5;
    if (pl->nbpotfire > 5)
        pl->nbpotfire = 5;
    sfText_setString(map->menu->text[23], numtostr(pl->nbpotfire, str));
    sfText_setString(map->menu->text[19], numtostr(pl->nbpothp, str));
    sfText_setString(map->menu->text[21], numtostr(pl->nbpotmana, str));
}