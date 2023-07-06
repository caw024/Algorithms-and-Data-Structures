#include <bits/stdc++.h>

#define debug(x) std::cout << #x << "=" << x << '\n'

using ll = long long;
ll MOD = 1e9 + 7;
ll INF = 1e18;

// use for unordered_map/unordered_set
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};


void solve() {
    int n;
    std::cin >> n;
    std::vector<int> arr(n, 0);
    for (int i = 0; i < n; i++){
        std::cin >> arr[i];
    }
}

int main() {
    std::freopen("input.txt", "r", stdin);
    std::ios::sync_with_stdio(0); std::cin.tie(0);

    int t;
    std::cin >> t;
    for (int i = 0; i < t; i++){
        solve();
    }
}

// // other way to parse input if no specified limit
// std::string s;
// std::ifstream if_file("input.txt");
// while (getline(if_file, s)){
//     // do something with s
// }

