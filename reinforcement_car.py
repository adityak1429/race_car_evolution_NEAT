import math
import random
import sys
import os

import neat
import pygame

# Screen dimensions
WIDTH = 1920
HEIGHT = 1080

# Car dimensions
CAR_SIZE_X = 60    
CAR_SIZE_Y = 60

# Border color to detect crashes
BORDER_COLOR = (0, 0, 0, 255) 
map='map_with_exit.png'
# Generation counter
current_generation = 0

class Car:

    def __init__(self):
        # Load and scale the car image
        self.car = pygame.image.load('car.png').convert()
        self.car = pygame.transform.scale(self.car, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_car = self.car

        # Initial position and properties
        self.position = [830, 920]
        self.angle = 0
        self.speed = 0

        # Flag to set default speed
        self.speed_set = False

        # Calculate center of the car
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

        # Lists to store radar information
        self.radars = []
        self.drawing_radars = []

        # Alive status
        self.alive = True

        # Track distance and time
        self.distance = 0
        self.time = 0

    def draw(self, screen):
        # Draw the car and optionally the radars
        screen.blit(self.rotated_car, self.position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        # Optionally draw all radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # Check for collisions with the border
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Extend radar until it hits the border or reaches max length
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calculate distance to the border and store it
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    
    def update(self, game_map):
        # Set initial speed if not set
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Rotate and move the car
        self.rotated_car = self.rotate_center(self.car, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Update distance and time
        self.distance += self.speed
        self.time += 1

        # Update Y-position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Recalculate the center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate the car's corners
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check for collisions and clear radars
        self.check_collision(game_map)
        self.radars.clear()

        # Update radars
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get distances to the border from the radars
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        # Check if the car is still alive
        return self.alive

    def get_reward(self):
        # Calculate and return the reward
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        # Rotate the image around its center
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image



def run_simulation(genomes, config):
    
    # Initialize collections for neural networks and cars
    nets = []
    cars = []
    # Create neural networks for each genome
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    # Setup clock and fonts, load the map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 20)
    alive_font = pygame.font.SysFont("Arial", 15)
    game_map = pygame.image.load(map).convert()
    global current_generation
    current_generation += 1

    # Simple counter to limit simulation time
    counter = 0

    while True:
        # Exit on quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # Get actions for each car from the neural networks
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10 # Turn left
            elif choice == 1:
                car.angle -= 10 # Turn right
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2 # Slow down
            else:
                car.speed += 2 # Speed up
        
        # Update and check if cars are still alive
        currently_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                currently_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if currently_alive == 0:
            break

        counter += 1
        if counter == 30 * 40: # Limit simulation to about 20 seconds
            break

        # Draw the map and cars
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        

        text = generation_font.render(f"Generation: {current_generation}", True,(255,255,255))
        text_rect = text.get_rect()
        text_rect.midtop = (WIDTH // 2, 10)  # Position at the top center with a 10-pixel margin
        screen.blit(text, text_rect)

        text = alive_font.render(f"Alive: {currently_alive}", True, (255,255,255))
        text_rect = text.get_rect()
        text_rect.midtop = (WIDTH // 2, 30)  # Position slightly below the first text
        screen.blit(text, text_rect)



        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    # Load configuration and run NEAT
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create population and add reporters
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # Initialize PyGame and the display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    # Run NEAT
    p.run(run_simulation, 1000)
