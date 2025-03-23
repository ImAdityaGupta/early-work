#include "const_setups.h"



BB nodes_checked = 0;
BB hash_exact = 0;
BB hash_alpha = 0;
BB hash_beta = 0;


int ms1bTable[256];
BB zobrist_table[64][9];
BB zobrist_switch_side;
mt19937 mt(01234567);

HASHE hash_table[HASH_LENGTH];



void init_ms1bTable() {
    int i;
    for (i=0; i<256; i++) {
        ms1bTable[i] = (
            (i>127) ? 7 :
            (i> 63) ? 6 :
            (i> 31) ? 5 :
            (i> 15) ? 4 :
            (i>  7) ? 3 :
            (i>  3) ? 2 :
            (i>  1) ? 1 :
                      0
        );
    }
}

BB randomInt() {
  BB r = 0;
  for (int i=0; i<64; i++) {
    r = r*2 + rand()%2;
  }
  return r;
}


void init_zobrist(){
    for (int i=0;i<64;i++){
        for (int j=0;j<9;j++){
            zobrist_table[i][j] = randomInt();
            //cout << i << " " << j << " " << bitset<64>(zobrist_table[i][j]) << endl;
        }
    }

    zobrist_switch_side = randomInt();

}
