#include<iostream>
#include<vector>
#include<cstdio>
using namespace std;



int coinChange(int coins[], int length,int amount){
    cout << "Length of coins:" << length << endl;
    for(int i=0;i<length;i++){
        cout << coins[i] << endl;
    }
    return 0;
}

int main(){
    int coins[] = {1,2,5};
    int length = sizeof(coins)/sizeof(coins[0]);
    int amount = 11;
    coinChange(coins,length,amount);
    return 0;
}