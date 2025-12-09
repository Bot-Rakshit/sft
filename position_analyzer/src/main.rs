use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use anyhow::Result;
use chess::{Board, ChessMove, MoveGen};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use indicatif::{ProgressBar, ProgressStyle};
use std::str::FromStr;

#[derive(Debug, Deserialize, Serialize)]
struct Position {
    fen: String,
    phase: String,
}

#[derive(Debug, Clone, Serialize)]
struct TopMove {
    r#move: String,
    eval_cp: i32,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct TrainingExample {
    messages: Vec<Message>,
}

fn count_material(board: &Board) -> i32 {
    let piece_values = [(chess::Piece::Pawn, 1), (chess::Piece::Knight, 3), 
                        (chess::Piece::Bishop, 3), (chess::Piece::Rook, 5), 
                        (chess::Piece::Queen, 9)];
    
    let stm = board.side_to_move();
    let mut stm_material = 0;
    let mut opp_material = 0;
    
    for (piece, value) in piece_values {
        stm_material += (board.pieces(piece) & board.color_combined(stm)).popcnt() as i32 * value;
        opp_material += (board.pieces(piece) & board.color_combined(!stm)).popcnt() as i32 * value;
    }
    
    stm_material - opp_material
}

fn analyze_position(fen: &str, stockfish_path: &str, depth: u8) -> Result<Vec<TopMove>> {
    let mut child = Command::new(stockfish_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    
    let stdin = child.stdin.as_mut().unwrap();
    stdin.write_all(b"uci\n")?;
    stdin.write_all(format!("position fen {}\n", fen).as_bytes())?;
    stdin.write_all(format!("go depth {} multipv 5\n", depth).as_bytes())?;
    
    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);
    
    let mut top_moves = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        
        if line.starts_with("bestmove") {
            break;
        }
        
        if line.contains("depth") && line.contains("multipv") && line.contains("score") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            
            if let (Some(pv_idx), Some(score_idx), Some(move_idx)) = (
                parts.iter().position(|&x| x == "multipv"),
                parts.iter().position(|&x| x == "score"),
                parts.iter().position(|&x| x == "pv")
            ) {
                if score_idx + 2 < parts.len() && move_idx + 1 < parts.len() {
                    let score_type = parts[score_idx + 1];
                    let score_val = parts[score_idx + 2];
                    let best_move = parts[move_idx + 1];
                    
                    let eval_cp = if score_type == "cp" {
                        score_val.parse::<i32>().unwrap_or(0)
                    } else if score_type == "mate" {
                        let mate_in = score_val.parse::<i32>().unwrap_or(0);
                        if mate_in > 0 { 10000 } else { -10000 }
                    } else {
                        0
                    };
                    
                    let multipv_num = parts.get(pv_idx + 1)
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(0);
                    
                    if multipv_num > 0 && multipv_num <= 5 {
                        if top_moves.len() < multipv_num {
                            top_moves.resize(multipv_num, TopMove {
                                r#move: String::new(),
                                eval_cp: 0,
                            });
                        }
                        top_moves[multipv_num - 1] = TopMove {
                            r#move: best_move.to_string(),
                            eval_cp,
                        };
                    }
                }
            }
        }
    }
    
    child.kill().ok();
    
    Ok(top_moves)
}

fn create_training_example(
    board: &Board,
    fen: &str,
    phase: &str,
    top_moves: &[TopMove],
) -> Result<TrainingExample> {
    let legal_moves: Vec<String> = MoveGen::new_legal(board)
        .map(|m| format!("{}", m))
        .collect();
    let legal_moves_str = legal_moves.join(" ");
    
    let material = count_material(board);
    let mobility = legal_moves.len();
    
    let top_moves_str: String = top_moves
        .iter()
        .map(|m| format!("{}:{}", m.r#move, m.eval_cp))
        .collect::<Vec<_>>()
        .join(" | ");
    
    let best_move = &top_moves[0].r#move;
    let best_eval = top_moves[0].eval_cp;
    
    let prompt = format!(
        "You are an expert chess player. Here is the position in FEN format:\n\
{}\n\n\
Legal moves: {}\n\n\
Position analysis:\n\
- Game phase: {}\n\
- Material advantage: {:+}\n\
- Mobility (legal moves): {}\n\
- Top moves with evaluations: {}\n\n\
Select the best move. Keep your thinking brief, then output your chosen move.\n\
Format:\n\
<think>brief analysis</think>\n\
<uci_move>your_move</uci_move>",
        fen, legal_moves_str, phase, material, mobility, top_moves_str
    );
    
    let response = format!(
        "<think>Best move {} with eval {:+}cp. Material {:+}, mobility {}.</think><uci_move>{}</uci_move>",
        best_move, best_eval, material, mobility, best_move
    );
    
    Ok(TrainingExample {
        messages: vec![
            Message {
                role: "user".to_string(),
                content: prompt,
            },
            Message {
                role: "assistant".to_string(),
                content: response,
            },
        ],
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 4 {
        eprintln!("Usage: {} <input_positions.jsonl> <output_training.jsonl> <stockfish_path> [depth]", args[0]);
        std::process::exit(1);
    }
    
    let input_file = &args[1];
    let output_file = &args[2];
    let stockfish_path = &args[3];
    let depth: u8 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(7);
    
    println!("Reading positions from: {}", input_file);
    println!("Output file: {}", output_file);
    println!("Stockfish path: {}", stockfish_path);
    println!("Analysis depth: {}", depth);
    
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);
    
    let positions: Vec<Position> = reader
        .lines()
        .filter_map(|line| line.ok())
        .filter_map(|line| serde_json::from_str(&line).ok())
        .collect();
    
    println!("Loaded {} positions", positions.len());
    println!("Starting parallel analysis using {} threads...\n", rayon::current_num_threads());
    
    let pb = ProgressBar::new(positions.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")?
            .progress_chars("#>-")
    );
    
    let output = Arc::new(Mutex::new(File::create(output_file)?));
    let errors = Arc::new(Mutex::new(0));
    
    positions.par_iter().for_each(|pos| {
        match Board::from_str(&pos.fen) {
            Ok(board) => {
                match analyze_position(&pos.fen, stockfish_path, depth) {
                    Ok(top_moves) if !top_moves.is_empty() => {
                        match create_training_example(&board, &pos.fen, &pos.phase, &top_moves) {
                            Ok(example) => {
                                if let Ok(json) = serde_json::to_string(&example) {
                                    let mut file = output.lock().unwrap();
                                    writeln!(file, "{}", json).ok();
                                }
                            }
                            Err(_) => {
                                *errors.lock().unwrap() += 1;
                            }
                        }
                    }
                    _ => {
                        *errors.lock().unwrap() += 1;
                    }
                }
            }
            Err(_) => {
                *errors.lock().unwrap() += 1;
            }
        }
        pb.inc(1);
    });
    
    pb.finish_with_message("Analysis complete!");
    
    let error_count = *errors.lock().unwrap();
    println!("\nDone! Errors: {}", error_count);
    println!("Training data written to: {}", output_file);
    
    Ok(())
}
