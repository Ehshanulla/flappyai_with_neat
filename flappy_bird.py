"""
The classic game of flappy bird. Make with python
and pygame. Features pixel perfect collision using masks :o

Date Modified:  Jul 30, 2019
Author: Tech With Tim
Estimated Work Time: 5 hours (1 just for that damn collision)
"""
import pygame
import random
import os
import time
import neat
import visualize
import pickle
pygame.font.init()  # init font

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

gen = 0

class Bird:
    """
    Bird class representing the flappy bird
    """
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """
        self.x = x
        self.y = y
        self.tilt = 0  # degrees to tilt
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        """
        make the bird jump
        :return: None
        """
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """
        make the bird move
        :return: None
        """
        self.tick_count += 1

        # for downward acceleration
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2  # calculate displacement

        # terminal velocity
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:  # tilt up
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:  # tilt down
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        """
        draw the bird
        :param win: pygame window or surface
        :return: None
        """
        self.img_count += 1

        # For animation of bird, loop through three images
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        # so when bird is nose diving it isn't flapping
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2


        # tilt the bird
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        """
        gets the mask for the current image of the bird
        :return: None
        """
        return pygame.mask.from_surface(self.img)


class Pipe():
    """
    represents a pipe object
    """
    GAP = 200
    VEL = 5

    def __init__(self, x):
        """
        initialize pipe object
        :param x: int
        :param y: int
        :return" None
        """
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img

        self.passed = False

        self.set_height()

    def set_height(self):
        """
        set the height of the pipe, from the top of the screen
        :return: None
        """
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        """
        move pipe based on vel
        :return: None
        """
        self.x -= self.VEL

    def draw(self, win):
        """
        draw both the top and bottom of the pipe
        :param win: pygame window/surface
        :return: None
        """
        # draw top
        win.blit(self.PIPE_TOP, (self.x, self.top))
        # draw bottom
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))


    def collide(self, bird, win):
        """
        returns if a point is colliding with the pipe
        :param bird: Bird object
        :return: Bool
        """
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)

        if b_point or t_point:
            return True

        return False

class Base:
    """
    Represnts the moving floor of the game
    """
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        """
        Initialize the object
        :param y: int
        :return: None
        """
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        """
        move floor so it looks like its scrolling
        :return: None
        """
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        """
        Draw the floor. This is two images that move together.
        :param win: the pygame surface/window
        :return: None
        """
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    """
    Rotate a surface and blit it to the window
    :param surf: the surface to blit to
    :param image: the image surface to rotate
    :param topLeft: the top left position of the image
    :param angle: a float value for angle
    :return: None
    """
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)

    surf.blit(rotated_image, new_rect.topleft)

def draw_window(win, birds, pipes, base, score, gen, pipe_ind):
    """
    draws the windows for the main game loop
    :param win: pygame window surface
    :param bird: a Bird object
    :param pipes: List of pipes
    :param score: score of the game (int)
    :param gen: current generation
    :param pipe_ind: index of closest pipe
    :return: None
    """
    if gen == 0:
        gen = 1
    win.blit(bg_img, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        # draw lines from bird to pipe
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        # draw bird
        bird.draw(win)

    # score
    score_label = STAT_FONT.render("Score: " + str(score),1,(255,255,255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    # generations
    score_label = STAT_FONT.render("Gens: " + str(gen-1),1,(255,255,255))
    win.blit(score_label, (10, 10))

    # alive
    score_label = STAT_FONT.render("Alive: " + str(len(birds)),1,(255,255,255))
    win.blit(score_label, (10, 50))

    pygame.display.update()


def eval_genomes(genomes, config):
    """
    Runs the simulation of the current population of birds and sets their fitness.
    Automatically saves the best genome when any bird reaches score 100.
    """
    global WIN, gen
    win = WIN
    gen += 1

    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230,350))
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0
    best_score = 0  # Track highest score achieved
    best_genome = None  # Store the best genome

    clock = pygame.time.Clock()
    run = True
    
    while run and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            bird.move()

            output = nets[birds.index(bird)].activate((bird.y, 
                                                     abs(bird.y - pipes[pipe_ind].height), 
                                                     abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            for bird in birds:
                if pipe.collide(bird, win):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            # Update best score and genome when score increases
            if score > best_score:
                best_score = score
                if birds:  # Check if any birds remain
                    best_genome = ge[birds.index(birds[0])]
            
            # Reward all surviving genomes
            for genome in ge:
                genome.fitness += 5
            
            pipes.append(Pipe(WIN_WIDTH))

            # Save immediately if target score reached
            if best_score >= 100 and best_genome:
                with open('winner.pkl', 'wb') as output:
                    pickle.dump(best_genome, output, 1)
                print(f"\n🎯 Saved winner with score: {best_score}")
                # Optional: Add visual feedback
                win_text = END_FONT.render("TARGET REACHED!", 1, (255,255,0))
                WIN.blit(win_text, (WIN_WIDTH//2 - win_text.get_width()//2, 100))
                pygame.display.update()
                pygame.time.delay(1000)  # Pause to show message
                run = False
                break

        for r in rem:
            pipes.remove(r)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)

    # Save best genome if training completes without reaching target
    if best_genome and best_score < 100:
        with open('winner.pkl', 'wb') as output:
            pickle.dump(best_genome, output, 1)
        print(f"\n🏆 Saved best genome with score: {best_score}")


def run(config_file, target_score=100, generations=50):
    """
    Runs the NEAT algorithm to train a neural network to play flappy bird.
    Automatically saves the best genome when any bird reaches the target score.
    
    Args:
        config_file (str): Path to NEAT config file
        target_score (int): Score at which to stop training and save winner (default: 100)
        generations (int): Maximum number of generations to run (default: 50)
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation,
                               config_file)

    # Create population
    p = neat.Population(config)
    
    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Track best genome globally
    global_best_genome = None
    global_best_score = 0
    
    # Custom reporter to track best score
    class BestScoreReporter(neat.reporting.BaseReporter):
        def post_evaluate(self, config, population, species, best_genome):
            nonlocal global_best_genome, global_best_score
            if best_genome.fitness > global_best_score:
                global_best_score = best_genome.fitness
                global_best_genome = best_genome
                
                # Save immediately if target reached
                if global_best_score >= target_score:
                    with open('winner.pkl', 'wb') as f:
                        pickle.dump(global_best_genome, f)
                    print(f"\n🔥 Target score {target_score} reached! Saved winner.pkl")
    
    p.add_reporter(BestScoreReporter())
    
    try:
        # Run NEAT
        winner = p.run(eval_genomes, generations)
        
        # Save best genome if target wasn't reached
        if global_best_score < target_score:
            with open('winner.pkl', 'wb') as f:
                pickle.dump(winner, f)
            print(f"\n🏆 Training completed. Best score: {global_best_score}. Saved winner.pkl")
            
        # Show final stats
        print('\nBest genome:\n{!s}'.format(winner))
        print(f'Best score achieved: {global_best_score}')
        
        # Run the winner
        run_winner(config, winner, score_limit=target_score)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
    finally:
        # Ensure winner is saved even if interrupted
        if global_best_genome and not os.path.exists('winner.pkl'):
            with open('winner.pkl', 'wb') as f:
                pickle.dump(global_best_genome, f)

def run_winner(config, genome, score_limit=100, show_visuals=True):
    """
    Run the game with the winning genome.
    
    Args:
        config (neat.Config): NEAT configuration object
        genome (neat.DefaultGenome): The winning genome to run
        score_limit (int): Score at which to stop and declare victory (default: 100)
        show_visuals (bool): Whether to display the pygame window (default: True)
    """
    # Initialize network and game objects
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    bird = Bird(230, 350)
    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0
    clock = pygame.time.Clock()
    
    # Performance tracking
    start_time = time.time()
    frames = 0
    run = True
    
    while run:
        frames += 1
        clock.tick(30)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                return
            
        # Game logic
        pipe_ind = 0 if len(pipes) <= 1 or bird.x <= pipes[0].x + pipes[0].PIPE_TOP.get_width() else 1
        
        # Neural network decision
        inputs = (
            bird.y, 
            abs(bird.y - pipes[pipe_ind].height), 
            abs(bird.y - pipes[pipe_ind].bottom)
        )
        output = net.activate(inputs)
        
        if output[0] > 0.5:
            bird.jump()
        
        # Update game state
        bird.move()
        base.move()
        
        # Pipe management
        add_pipe = False
        for pipe in pipes[:]:
            pipe.move()
            
            if pipe.collide(bird, WIN):
                run = False
                break
                
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                pipes.remove(pipe)
                
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True
        
        if add_pipe:
            score += 1
            pipes.append(Pipe(WIN_WIDTH))
        
        # Check termination conditions
        if (bird.y + bird.img.get_height() - 10 >= FLOOR or 
            bird.y < -50):
            run = False
            
        # Check score limit
        if score >= score_limit:
            if show_visuals:
                win_text = END_FONT.render("WINNER!", 1, (255, 255, 0))
                WIN.blit(win_text, (WIN_WIDTH//2 - win_text.get_width()//2, 
                                   WIN_HEIGHT//2 - win_text.get_height()//2))
                pygame.display.update()
                pygame.time.delay(2000)
            run = False
        
        # Display if enabled
        if show_visuals:
            draw_window(WIN, [bird], pipes, base, score, 0, pipe_ind)
        else:
            # Print progress for headless mode
            if frames % 30 == 0:
                print(f"Score: {score}, Time: {time.time()-start_time:.1f}s")
    
    # Performance stats
    duration = time.time() - start_time
    fps = frames / duration if duration > 0 else 0
    print(f"\nFinal Score: {score}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Average FPS: {fps:.1f}")
    
    if show_visuals:
        pygame.quit()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_score', type=int, default=100,
                      help='Score to aim for during training')
    parser.add_argument('--generations', type=int, default=50,
                      help='Maximum generations to run')
    args = parser.parse_args()
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    
    run(config_path, target_score=args.target_score, generations=args.generations)
