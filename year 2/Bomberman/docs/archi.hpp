#include <string>
#include <vector>
#include <raylib.h>

/*                          ATTENTION /!\ La Lib Graphique considère le clavier comme Qwerty                          */

#define SAVING_PATH "./save/"   /*      Sauvegarde Path                             */
#define BOMB_TIMER 3            /*      Time in seconds before bombs explodes       */

namespace Enum {                            /*      Namespace Enum Begin        */
    enum MAP    {BLOCK, BREAK, BREAKT, SPACE};  /*      Enum MAP : Naming all used block in the game on the map         */
    enum DIREC  {LEFT, RIGHT, DOWN, UP};        /*      Enum DIREC : Naming all diferent directions for player movement */
    enum POWER  {BOMB, SPEED, FIRE, WALL};      /*      Enum POWER : Naming all the power up that the player can have   */
    enum BOMB   {INIT, MID, PATH, END};         /*      Enum BOMB : Naming all part of the bomb exploded and not        */
}                                           /*      Namespace Enum End          */

namespace Game {                            /*      Namespace Game Begin        */
    class Player {                                  /*      Class Player Begin      */
        public:
            Player();
            std::vector<KeyboardKey> getKeys() const;
            int Move(Enum::DIREC dir, std::vector<std::vector<int>> map);
            void powerUp(Enum::POWER p);
            void createBomb();
            bool checkBombs();
            Bomb *getExplodingBomb();
            ~Player() = default;

        private:
            bool _bombUp;
            bool _speedUp;
            bool _fireUp;
            bool _wallPass;

            Vector2 _pos;
            std::vector<KeyboardKey> _dir[4];
            std::vector<Bomb*> _bombs;
            KeyboardKey _bombIt;
            std::string _name;

    };                                              /*      Class Player End        */
    class Bomb {                                    /*      Class Bomb Begin        */
        public:
            Bomb();
            void countdown();
            bool explosion();
            Vector2 getPos() const;
            int getRange();
            ~Bomb() = default;

        private:
            int _range;
            Vector2 _pos;
            bool _explosion;
    };                                              /*      Class Bomb End          */
}                                           /*      Namespace Game End          */

namespace Graph {                           /*      Namespace Graph Begin       */
    class Window {                                  /*      Class Window Begin      */
        public:
            Window(Vector2 size, std::string name);
            void open();
            void clear();
            void display();
            void close();
            ~Window() = default;

        private:
            Vector2 _size;
    };                                              /*      Class Window End        */
    class Text {
        public:
            Text();
            void draw(Vector2 move);
            Vector2 getSize() const;
            Vector2 getPos() const;
            ~Text();

        private:
            Vector2 _size;
            Vector2 _pos;
    };
    class Button {                                  /*      Class Button Begin      */
        public:
            Button();
            ~Button();
    };                                              /*      Class Button End        */
}                                           /*      Namespace Graph End        */