#include<cstdio>
#include<iostream>
#include<map>
#include<vector>
using namespace std;


long long int fibi(int N){
    if(N==1 || N==2){
        return 1;
    }
    return fibi(N-1) + fibi(N-2);
}


long long int helper(int n,vector<int>& memo){
    //base case
    if(n==1 || n==2) return 1;
    //已经计算过了
    if(memo[n] !=0) return memo[n];
    memo[n] = helper(n-1,memo) + helper(n-2,memo);
    return memo[n];
}

long long int fibi_v1(int N){
    if(N<1) return 0;
    vector<int> record(N,0);
    return helper(N,record);
}

long long int fibi_v2(int N){
    //这个才是最优解法,时间复杂度O(N) 空间复杂度O(1)
    if(N<1) return 0;
    long long int a = 0;
    long long int b = 1;
    long long int c;
    for(int i = 1;i<N;i++){
        c = a+b;
        a = b;
        b = c;
    }
    return c;
}

long long int fibi_v3(int N){
    //时间复杂度O(N) 空间复杂度O(N)
    if(N<1) return 0;
    vector<long int> dp(N,0);
    dp[0] = 0;
    dp[1] = 1;
    for(int i=2;i<=N;i++){
        dp[i] = dp[i-1] + dp[i-2];
    }
    return dp[N];
}



int main(){
    cout << "hello,fibi" << endl;

    long long int value;
    int N = 20;
//    value = fibi(N);
//    value = fibi_v1(N);
    value = fibi_v2(N);
//    value = fibi_v3(N);
    cout << "value: " << value << endl;
    return 0;
}