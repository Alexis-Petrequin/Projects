/*
** EPITECH PROJECT, 2020
** init_other
** File description:
** about usefull functions
*/

#include "../../include/mylist.h"

sfRenderWindow *fill_mywindow(int width, int height, char const *name)
{
    sfVideoMode video_mode = {width, height, 32};

    return (sfRenderWindow_create(video_mode, name, sfDefaultStyle, NULL));
}

player_t *init_player(void)
{
    player_t *player = malloc(sizeof(player_t));

    if (!player)
        return (NULL);
    player->state = 0;
    player->mvol = 70;
    player->svol = 100;
    player->state = 0;
    player->level = 1;
    player->nbpotfire = 1;
    player->nbpothp = 1;
    player->nbpotmana = 1;
    player->monsterkilled = 0;
    player->quest = 0;
    return (player);
}

sfClock **init_clock(int nb)
{
    sfClock **clock = malloc(sizeof(sfClock *) * (nb + 1));

    for (int i = 0; i < nb; i++)
        clock[i] = sfClock_create();
    clock[nb] = NULL;
    return (clock);
}

sprite_t *make_sprite(sfTexture *texture, sfvector4f_t posize, sfVector2f sp)
{
    sprite_t *sprite = malloc(sizeof(sprite_t));

    if (!sprite)
        return (NULL);
    sprite->sprite = sfSprite_create();
    sprite->pos.x = posize.x;
    sprite->pos.y = posize.y;
    sprite->scale.x = posize.a;
    sprite->scale.y = posize.b;
    sprite->speed.x = sp.x;
    sprite->speed.y = sp.y;
    sfSprite_setTexture(sprite->sprite, texture, sfFalse);
    sfSprite_setPosition(sprite->sprite, sprite->pos);
    sfSprite_scale(sprite->sprite, sprite->scale);
    return (sprite);
}