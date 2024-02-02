/*
** EPITECH PROJECT, 2020
** map_init
** File description:
** about init
*/

#include "../../include/mylist.h"

static sfTexture **initexture_map(void)
{
    sfTexture **text = malloc(sizeof(sfTexture *) * 12);

    if (!text)
        return (NULL);
    text[0] = sfTexture_createFromFile("assets/maplvl/tree.png", NULL);
    text[1] = sfTexture_createFromFile("assets/maplvl/house.png", NULL);
    text[2] = sfTexture_createFromFile("assets/maplvl/house1.png", NULL);
    text[3] = sfTexture_createFromFile("assets/maplvl/house2.png", NULL);
    text[4] = sfTexture_createFromFile("assets/maplvl/blacksmith.png", NULL);
    text[5] = sfTexture_createFromFile("assets/maplvl/hunter.png", NULL);
    text[6] = sfTexture_createFromFile("assets/maplvl/sage.png", NULL);
    text[7] = sfTexture_createFromFile("assets/maplvl/popup_1.png", NULL);
    text[8] = sfTexture_createFromFile("assets/maplvl/map.png", NULL);
    text[9] = sfTexture_createFromFile("assets/maplvl/walk.png", NULL);
    text[10] = sfTexture_createFromFile("assets/button.png", NULL);
    text[11] = NULL;
    return (text);
}

static perso_t *init_perso(map_t *map)
{
    perso_t *perso = malloc(sizeof(perso_t));

    if (!perso)
        return (NULL);
    perso->perso = sfSprite_create();
    perso->map = sfSprite_create();
    sfSprite_setTexture(perso->map, map->texture[8], sfTrue);
    sfSprite_setTexture(perso->perso, map->texture[9], sfTrue);
    sfSprite_setScale(perso->map, (sfVector2f){ 1.7, 1.7 });
    sfSprite_setScale(perso->perso, (sfVector2f){ 0.25, 0.25 });
    perso->rect.height = 516;
    perso->rect.left = 0;
    perso->rect.top = 516;
    perso->rect.width = 484;
    sfSprite_setTextureRect(perso->perso, perso->rect);
    perso->pos.x = sfView_getCenter(map->camera).x - 45;
    perso->pos.y = sfView_getCenter(map->camera).y - 45;
    sfSprite_setPosition(perso->perso, perso->pos);
    return (perso);
}

static sprit_t **fill_pnj(sfTexture **texture)
{
    sprit_t **pnj = malloc(sizeof(sprit_t *) * 4);
    sfVector2f pos[4] = { {9470, 8225}, {9020, 10150}, {8950, 7300},
                        {0.125, 0.125} };

    if (!pnj)
        return (NULL);
    for (int i = 0; i < 3; ++i) {
        pnj[i] = malloc(sizeof(sprit_t));
        if (!pnj[i])
            return (NULL);
        pnj[i]->sprite = sfSprite_create();
        sfSprite_setTexture(pnj[i]->sprite, texture[i + 4], sfTrue);
        sfSprite_setPosition(pnj[i]->sprite, pos[i]);
        sfSprite_setScale(pnj[i]->sprite, pos[3]);
    }
    pnj[3] = NULL;
    return (pnj);
}

static button_t **init_dial(map_t *map)
{
    button_t **dial = malloc(sizeof(button_t *) * 4);

    if (!dial)
        return (NULL);
    for (int y = 0; y < 3; ++y) {
        dial[y] = malloc(sizeof(button_t) * 4);
        if (!dial[y])
            return (NULL);
        for (int i = 0; i < 3; ++i) {
            dial[y][i].sprite = sfRectangleShape_create();
            dial[y][i].text = sfText_create();
            fill_dial(dial, y, i, map);
        }
    }
    dial[3] = NULL;
    return (dial);
}

void init_map2(map_t *map)
{
    map->tuto = 0;
    sfText_setString(map->presse, "PRESS E");
    sfText_setFont(map->presse, map->font);
    map->interface = init_interface(map);
    map->sound = sfSound_create();
    map->s_buffer = sfSoundBuffer_createFromFile("assets/m_sound.ogg");
    sfSound_setBuffer(map->sound, map->s_buffer);
    map->music = sfMusic_createFromFile("assets/map_music.ogg");
}

void init_menu_map2(sfFont *font, sfText **text)
{
    char str[24][30] = { {"STATUS"}, {"STATS"}, {"PV"}, {"100"}, {"MANA"},
    {"100"}, {"FORCE"}, {"10"}, {"DEFENSE"}, {"10"}, {"MAGIC"}, {"10"},
    {"LEVEL"}, {"10"}, {"QUEST"}, {"Kill 10 monsters :"}, {"10"},
    {"Inventory"}, {"heal potion :"}, {"2"}, {"mana potion :"}, {"2"},
    {"fire potion :"}, {"2"} };

    for (int i = 0; i < 24; i++) {
        text[i] = sfText_create();
        sfText_setFont(text[i], font);
        sfText_setString(text[i], str[i]);
        sfText_setPosition(text[i], (sfVector2f){ -1000, -1000 });
    }
    text[24] = NULL;
}

tab_t *init_menu_map(sfFont *font)
{
    tab_t *menu = malloc(sizeof(tab_t));
    sfColor color = sfBlack;
    color.a = 140;

    if (!menu)
        return (NULL);
    menu->rectangle = malloc(sizeof(sfRectangleShape *) * 3);
    menu->text = malloc(sizeof(sfText *) * 25);
    if (!menu->rectangle || !menu->text)
        return (NULL);
    for (int i = 0; i < 2; i++) {
        menu->rectangle[i] = sfRectangleShape_create();
        sfRectangleShape_setFillColor(menu->rectangle[i], color);
        sfRectangleShape_setSize(menu->rectangle[i], (sfVector2f){1200, 800});
        sfRectangleShape_setPosition(menu->rectangle[i],
        (sfVector2f){ -1000, -1000 });
    }
    menu->rectangle[2] = NULL;
    init_menu_map2(font, menu->text);
    return (menu);
}

map_t *init_map(void)
{
    map_t *map = malloc(sizeof(map_t));

    if (!map)
        return (NULL);
    map->event = malloc(sizeof(sfEvent));
    map->camera = sfView_create();
    sfView_reset(map->camera, (sfFloatRect){0, 0, 1920, 1026});
    sfView_setCenter(map->camera, (sfVector2f){8525, 6500});
    map->texture = initexture_map();
    if (!map->texture)
        return (NULL);
    map->perso = init_perso(map);
    map->scenary = init_scenary(map->texture);
    map->pnj = fill_pnj(map->texture);
    map->font = sfFont_createFromFile("assets/maplvl/RING.ttf");
    map->bubble = init_dial(map);
    map->presse = sfText_create();
    map->clock = init_clock(2);
    map->menu = init_menu_map(map->font);
    init_map2(map);
    return (map);
}