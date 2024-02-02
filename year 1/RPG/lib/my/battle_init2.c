/*
** EPITECH PROJECT, 2020
** battle_init2
** File description:
** about init battle
*/

#include "../../include/mylist.h"

void init_textu_battle2(sfTexture **texture)
{
    texture[0] = sfTexture_createFromFile("assets/blvl/buton.png", NULL);
    texture[1] = sfTexture_createFromFile("assets/blvl/2.png", NULL);
    texture[2] = sfTexture_createFromFile("assets/blvl/3.png", NULL);
    texture[3] = sfTexture_createFromFile("assets/blvl/4.png", NULL);
    texture[4] = sfTexture_createFromFile("assets/blvl/5.png", NULL);
    texture[5] = sfTexture_createFromFile("assets/blvl/6.png", NULL);
    texture[6] = sfTexture_createFromFile("assets/blvl/7.png", NULL);
    texture[7] = sfTexture_createFromFile("assets/blvl/8.png", NULL);
    texture[8] = sfTexture_createFromFile("assets/blvl/9.png", NULL);
    texture[9] = sfTexture_createFromFile("assets/blvl/10.png", NULL);
    texture[10] = sfTexture_createFromFile("assets/blvl/11.png", NULL);
    texture[11] = sfTexture_createFromFile("assets/blvl/battle.png", NULL);
    texture[12] = sfTexture_createFromFile("assets/blvl/orc3.png", NULL);
    texture[13] = sfTexture_createFromFile("assets/blvl/orc2.png", NULL);
    texture[14] = sfTexture_createFromFile("assets/blvl/orc.png", NULL);
    texture[15] = sfTexture_createFromFile("assets/blvl/1.png", NULL);
    texture[16] = NULL;
}

sfTexture **init_textu_battle(void)
{
    sfTexture **texture = malloc(sizeof(sfTexture *) * 17);

    if (!texture)
        return (NULL);
    init_textu_battle2(texture);
    for (int i = 0; i <= 15; i++)
        if (!texture[i])
            return (NULL);
    return (texture);
}

ressource_t *init_ress_battle(void)
{
    ressource_t *ress = malloc(sizeof(ressource_t));

    if (!ress)
        return (NULL);
    ress->texture = init_textu_battle();
    ress->police = sfFont_createFromFile("assets/albam.TTF");
    ress->s_buffer = sfSoundBuffer_createFromFile("assets/m_sound.ogg");
    ress->music = sfMusic_createFromFile("assets/blvl/battle.ogg");
    return (ress);
}

mob_t *make_mob(sfIntRect rect, sfVector2f pos, sfTexture *texture)
{
    mob_t *mob = malloc(sizeof(mob_t));

    mob->sprite = sfSprite_create();
    mob->rect = rect;
    sfSprite_setTexture(mob->sprite, texture, sfFalse);
    sfSprite_setTextureRect(mob->sprite, rect);
    sfSprite_setPosition(mob->sprite, pos);
    return (mob);
}

mob_t **init_mob(sfTexture **texture)
{
    mob_t **mob = malloc(sizeof(mob_t *) * 3);
    sfIntRect rect[2] = { {0, 1452, 363, 363}, {0, 0, 450, 450} };
    sfVector2f pos[2] = { {800, 630}, {100, 530} };
    int nb[2] = { 11, 12 };

    for (int i = 0; i < 2; i++)
        mob[i] = make_mob(rect[i], pos[i], texture[nb[i]]);
    mob[2] = NULL;
    return (mob);
}