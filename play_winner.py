import pickle
import pygame
import neat
import os
import sys
from flappy_bird import Bird, Pipe, Base, FLOOR, WIN_WIDTH, WIN_HEIGHT, WIN, draw_window

def run_winner(config, genome, score_limit=100):
    """Run the game with the winning genome."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    bird = Bird(230, 350)
    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0
    
    clock = pygame.time.Clock()
    run = True
    
    while run:
        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()
        
        pipe_ind = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1
        
        # Neural network decision
        output = net.activate((bird.y, 
                             abs(bird.y - pipes[pipe_ind].height), 
                             abs(bird.y - pipes[pipe_ind].bottom)))
        
        if output[0] > 0.5:
            bird.jump()
        
        # Game updates
        bird.move()
        base.move()
        
        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            
            if pipe.collide(bird, WIN):
                run = False
                break
                
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
                
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True
        
        if add_pipe:
            score += 1
            pipes.append(Pipe(WIN_WIDTH))
        
        for r in rem:
            pipes.remove(r)
        
        # Check for failure
        if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
            run = False
            
        # Check score limit
        if score >= score_limit:
            print(f"AI reached the score limit of {score_limit}!")
            run = False
        
        draw_window(WIN, [bird], pipes, base, score, 0, pipe_ind)

def main(config_path, genome_path, score_limit=100):
    """Main function to load and run the winner."""
    try:
        # Load NEAT config
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)
        
        # Verify files exist
        if not os.path.exists(genome_path):
            raise FileNotFoundError(f"Genome file not found at {genome_path}")
            
        # Load winner genome
        with open(genome_path, 'rb') as f:
            genome = pickle.load(f)
        
        # Run the winner
        run_winner(config, genome, score_limit)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()

if __name__ == '__main__':
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run the best Flappy Bird AI')
    parser.add_argument('--config', type=str, default='config-feedforward.txt',
                      help='Path to NEAT config file')
    parser.add_argument('--genome', type=str, default='winner.pkl',
                      help='Path to saved genome')
    parser.add_argument('--score_limit', type=int, default=100,
                      help='Score limit to stop the game')
    
    args = parser.parse_args()
    
    # Get absolute paths
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, args.config)
    genome_path = os.path.join(local_dir, args.genome)
    
    # Run the AI
    main(config_path, genome_path, args.score_limit)