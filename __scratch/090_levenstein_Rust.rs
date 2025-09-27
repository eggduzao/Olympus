use std::io::{self, BufRead};

fn levenshtein(s1: &str, s2: &str) -> usize {
    let m = s1.len();
    let n = s2.len();
    let mut dp = vec![vec![0; n + 1]; m + 1];
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
            let del = dp[i - 1][j] + 1;
            let ins = dp[i][j - 1] + 1;
            let sub = dp[i - 1][j - 1] + cost;
            dp[i][j] = del.min(ins.min(sub));
        }
    }

    dp[m][n]
}

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    let s1 = lines.next().unwrap().unwrap();
    let s2 = lines.next().unwrap().unwrap();

    println!("{}", levenshtein(&s1, &s2));
}

