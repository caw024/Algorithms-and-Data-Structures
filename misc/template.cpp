#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
ll MOD = 1e9 + 7;
ll INF = 1e12;


void initial_setup(){
    freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);

    // desync cin/scanf
    // removes auto-flush from calling cin then cout
    ios::sync_with_stdio(0); cin.tie(0);
}

void solve() {

}

int main() {
    initial_setup();

    int n, s, d;
    cin >> n >> s >> d;
    
    int tc = 1;
    // cin >> tc;
    for (int t = 1; t <= tc; t++) {
        // cout << "Case #" << t  << ": ";
        solve();
    }
}