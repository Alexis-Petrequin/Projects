/*
** EPITECH PROJECT, 2020
** menu_init2
** File description:
** about init menu
*/

#include "../../include/mylist.h"

sfTexture **init_textu_menu(void)
{
    sfTexture **texture = malloc(sizeof(sfTexture *) * 8);

    if (!texture)
        return (NULL);
    texture[0] = sfTexture_createFromFile("assets/button.png", NULL);
    texture[1] = sfTexture_createFromFile("assets/mlvl/sky.png", NULL);
    texture[2] = sfTexture_createFromFile("assets/mlvl/rocks_1.png", NULL);
    texture[3] = sfTexture_createFromFile("assets/mlvl/rocks_2.png", NULL);
    texture[4] = sfTexture_createFromFile("assets/mlvl/clouds_1.png", NULL);
    texture[5] = sfTexture_createFromFile("assets/mlvl/clouds_2.png", NULL);
    texture[6] = sfTexture_createFromFile("assets/mlvl/clouds_3.png", NULL);
    texture[7] = NULL;
    for (int i = 0; i <= 6; i++)
        if (!texture[i])
            return (NULL);
    return (texture);
}

ressource_t *init_ress_menu(void)
{
    ressource_t *ress = malloc(sizeof(ressource_t));

    if (!ress)
        return (NULL);
    ress->texture = init_textu_menu();
    ress->police = sfFont_createFromFile("assets/albam.TTF");
    ress->music = sfMusic_createFromFile("assets/mlvl/menu.ogg");
    ress->s_buffer = sfSoundBuffer_createFromFile("assets/m_sound.ogg");
    if (!ress->police || !ress->texture || !ress->music || !ress->s_buffer)
        return (NULL);
    return (ress);
}