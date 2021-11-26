#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
ll MOD = 1e9 + 7;
ll INF = 1e12;


void redirect_io(){
    ifstream in("input.txt");
    streambuf *cinbuf = cin.rdbuf(); //save old buf
    cin.rdbuf(in.rdbuf()); //redirect cin to input.txt!

    // ofstream out("output.txt");
    // streambuf *coutbuf = cout.rdbuf(); //save old buf
    // cout.rdbuf(out.rdbuf()); //redirect cout to output.txt!
}

void solve() {

}

int main() {
    redirect_io();

    ios_base::sync_with_stdio(false), cin.tie(NULL);

    int n, s, d;
    cin >> n >> s >> d;
    
    int tc = 1;
    // cin >> tc;
    for (int t = 1; t <= tc; t++) {
        // cout << "Case #" << t  << ": ";
        solve();
    }
}