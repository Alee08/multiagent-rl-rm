import pygame
import cv2
import os
import imageio


class EnvironmentRenderer:
    def __init__(self, grid_width, grid_height, agents, object_positions, goals):
        """Initialize renderer with grid size, agents, objects, and goal mapping."""
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.agents = agents
        self.object_positions = object_positions or {}  # Positions of in-world objects
        self.enable_ice_background = bool(
            self.object_positions.get("use_ice_background", False)
        )
        self.show_rm_panel = bool(self.object_positions.get("show_rm_panel", False))
        self.panel_width = 260  # Extra space on the right for overlays (e.g., RM)
        self.goals = goals
        self.frames = []
        self.agent_images = {}  # Cache for loaded agent images
        self.object_images = {}  # Mapping of object images
        self.resources = {}  # Mapping of loaded resources
        # Build the base path for image assets
        self.img_path = os.path.join(os.path.dirname(__file__))
        self.init_pygame()

    def load_resource(self, path, size):
        """Load an image from disk and scale it to the desired size."""
        # Use the image folder path to locate the asset
        full_path = os.path.join(self.img_path, path)
        image = pygame.image.load(full_path)
        return pygame.transform.scale(image, size)

    def init_pygame(self):
        """Set up pygame surfaces, fonts, and preload static resources."""
        pygame.init()
        self.cell_size = 100
        self.frames = []
        self.font = pygame.font.SysFont("Arial", 25)  # Create a font object

        # Resource dictionary with file paths and sizes
        resource_info = {
            "ita_man": ("img/ita_man.png", (85, 85)),
            "bcn_man": ("img/bcn_man2.png", (80, 80)),
            "CR7": ("img/CR7.png", (70, 70)),
            "juve": ("img/juve.png", (75, 75)),
            "holes": ("img/hole.png", (90, 90)),
            "ice": ("img/ice.png", (self.cell_size, self.cell_size)),
            "ponte_immagine": ("img/ponte_.png", (40, 40)),
            "barca_a_remi": ("img/barca_.png", (40, 40)),
            "plant": ("img/pianta.png", (90, 90)),  # Add the plant image
            "coffee": ("img/coffee.png", (90, 90)),  # Add the coffee image
            "letter": ("img/email.png", (90, 90)),  # Add the letter image
        }

        # Load all resources defined in resource_info
        for name, (path, size) in resource_info.items():
            self.resources[name] = self.load_resource(path, size)

        # Configure the pygame window
        screen_width = self.grid_width * self.cell_size + self.panel_width
        screen_height = self.grid_height * self.cell_size
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()

    def load_agent_image(self, agent_type):
        """Load and cache the sprite for a given agent type."""
        # Map agent types to their sprite paths and sizes
        image_map = {
            "a1": ("img/ita_man.png", (85, 85)),
            "a2": ("img/juve.png", (75, 75)),
            "a3": ("img/bcn_man2.png", (80, 80)),
            "a4": ("img/CR7.png", (80, 80)),
            "a5": ("img/juve.png", (80, 80)),
            # Add additional mappings as needed...
        }

        # Default sprite path and size
        default_image_path = "img/bcn_man2.png"
        default_image_size = (80, 80)

        # Check whether the sprite has already been cached
        if agent_type not in self.agent_images:
            try:
                # Load the agent-specific sprite or fall back to the default
                if agent_type in image_map:
                    file_name, size = image_map[agent_type]
                else:
                    # Use default sprite
                    file_name, size = default_image_path, default_image_size

                # Build the absolute asset path
                full_path = os.path.join(self.img_path, file_name)
                image = pygame.image.load(full_path)
                self.agent_images[agent_type] = pygame.transform.scale(image, size)

            except pygame.error as e:
                print(f"Error loading sprite for '{agent_type}': {e}")
                self.agent_images[agent_type] = None  # Or another default image

        return self.agent_images[agent_type]

    def get_agent_image(self, agent_name, small=False):
        """Return an agent sprite, optionally scaled down when multiple agents share a cell."""
        agent_type = agent_name  # Assume agent_name matches the type
        image = self.load_agent_image(agent_type)
        if image and small:
            return pygame.transform.scale(
                image, (image.get_width() // 2, image.get_height() // 2)
            )
        return image

    def render(self, episode, obs):
        """Draw the current environment state and collect frames for export."""
        cell_size = 100
        # Tune the update speed for early training versus final episodes; fallback for tags
        if isinstance(episode, int):
            self.clock.tick(6000 if episode < 89998 else 60)
        else:
            self.clock.tick(60)
        self.screen.fill((255, 255, 255))

        # Draw ice tiles as background for frozen lake cells
        if self.enable_ice_background:
            ice_image = self.resources.get("ice")
            if ice_image:
                for x in range(self.grid_width):
                    for y in range(self.grid_height):
                        self.screen.blit(ice_image, (x * cell_size, y * cell_size))

        # Draw the grid lines above the ice tiles
        for x in range(0, self.grid_width * cell_size, cell_size):
            for y in range(0, self.grid_height * cell_size, cell_size):
                colore_linea = (0, 0, 0)  # Black for all cells

                # Draw vertical and horizontal edges
                pygame.draw.line(self.screen, colore_linea, (x, y), (x, y + cell_size))
                pygame.draw.line(self.screen, colore_linea, (x, y), (x + cell_size, y))

        # Render plants
        for p_x, p_y in self.object_positions.get("plant", []):
            plant_image = self.resources.get("plant")
            if plant_image:
                self.screen.blit(plant_image, (p_x * 101, p_y * 101))

        # Render coffee
        for c_x, c_y in self.object_positions.get("coffee", []):
            coffee_image = self.resources.get("coffee")
            if coffee_image:
                self.screen.blit(coffee_image, (c_x * 101, c_y * 101))

        # Render letters
        for l_x, l_y in self.object_positions.get("letter", []):
            letter_image = self.resources.get("letter")
            if letter_image:
                self.screen.blit(letter_image, (l_x * 101, l_y * 101))

        # Draw obstacles
        for pos in self.object_positions.get("obstacles", []):
            obstacle_rect = pygame.Rect(
                pos[0] * self.cell_size,
                pos[1] * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(
                self.screen, (200, 90, 90), obstacle_rect
            )  # Red for obstacles

        # Draw holes
        for h_x, h_y in self.object_positions.get("holes", []):
            hole_image = self.resources["holes"]  # Grab the hole image from resources
            if hole_image:
                self.screen.blit(hole_image, (h_x * 101, h_y * 101))

        # Draw goals
        for goal_char, (g_x, g_y) in self.goals.items():
            goal_rect = pygame.Rect(
                g_x * self.cell_size,
                g_y * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(self.screen, (255, 215, 0), goal_rect)
            goal_text = self.font.render(goal_char, True, (0, 0, 0))
            text_rect = goal_text.get_rect(center=goal_rect.center)
            self.screen.blit(goal_text, text_rect)

        # Render office walls
        for (cell1, cell2) in self.object_positions.get("office_walls", []):
            x1, y1 = cell1
            x2, y2 = cell2
            if x1 == x2:  # Vertical wall
                start_pos = (x1 * cell_size, min(y1, y2) * cell_size + cell_size)
                end_pos = ((x1 + 1) * cell_size, min(y1, y2) * cell_size + cell_size)
            elif y1 == y2:  # Horizontal wall
                start_pos = (min(x1, x2) * cell_size + cell_size, y1 * cell_size)
                end_pos = (min(x1, x2) * cell_size + cell_size, (y1 + 1) * cell_size)
            pygame.draw.line(self.screen, (0, 0, 0), start_pos, end_pos, 8)

        # Track where agents currently reside
        agent_positions = {}

        # Gather position information for each agent
        for agent_name, agent_state in obs.items():
            pos_x = agent_state.get("pos_x", None)
            pos_y = agent_state.get("pos_y", None)
            if pos_x is not None and pos_y is not None:
                position = (pos_x, pos_y)
                if position not in agent_positions:
                    agent_positions[position] = []
                agent_positions[position].append(agent_name)
            else:
                # Handle missing positions gracefully
                print("Agent positions are not correctly defined!")

        # Draw agents
        for position, agents_at_pos in agent_positions.items():
            if len(agents_at_pos) > 1:
                # Multiple agents share the same cell: shrink and offset them
                for index, ag in enumerate(agents_at_pos):
                    agent_image = self.get_agent_image(
                        ag, small=True
                    )  # Get the scaled-down sprite
                    offset = (
                        index * cell_size // len(agents_at_pos),
                        0,
                    )  # Compute side-by-side offset
                    self.screen.blit(
                        agent_image,
                        (
                            position[0] * cell_size + offset[0],
                            position[1] * cell_size + offset[1],
                        ),
                    )
            else:
                # Single agent: use the full-size sprite
                # Offset to center within the cell
                ag = agents_at_pos[0]
                agent_image = self.get_agent_image(ag, small=False)
                image_width, image_height = (
                    agent_image.get_width(),
                    agent_image.get_height(),
                )
                base_x = position[0] * cell_size + (cell_size - image_width) // 2
                base_y = position[1] * cell_size + (cell_size - image_height) // 2
                self.screen.blit(agent_image, (base_x, base_y))

        # Overlay: show Reward Machine state per agent on the right panel
        self.render_rm_panel(obs)
        pygame.display.flip()

        # if episode % 5000 == 0:
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        image_data = image_data.transpose([1, 0, 2])
        self.frames.append(image_data)

    def render_rm_panel(self, obs):
        """Draw Reward Machine states for each agent on a side panel."""
        if not self.show_rm_panel:
            return

        panel_x = self.grid_width * self.cell_size + 10
        panel_left = self.grid_width * self.cell_size
        panel_top = 0
        panel_height = self.grid_height * self.cell_size

        # Panel background
        bg_rect = pygame.Rect(panel_left, panel_top, self.panel_width, panel_height)
        pygame.draw.rect(self.screen, (245, 245, 245), bg_rect)

        y = 10

        title = self.font.render("Reward Machine", True, (20, 20, 20))
        self.screen.blit(title, (panel_x, y))
        y += title.get_height() + 10

        has_rm = False
        for ag in self.agents:
            rm = getattr(ag, "reward_machine", None)
            if not rm:
                continue
            has_rm = True
            # Resolve steps source lazily if not provided
            steps_val = getattr(self, "agent_steps_view", None)
            if steps_val is None:
                env_ref = getattr(ag, "ma_problem", None)
                steps_val = getattr(env_ref, "agent_steps", None) if env_ref else None
            current_state = rm.get_current_state()
            final_state = rm.get_final_state()
            states_ordered = list(rm.state_indices.keys())
            label = f"{ag.name}"
            txt = self.font.render(label, True, (0, 0, 0))
            self.screen.blit(txt, (panel_x, y))
            y += txt.get_height() + 2

            # Steps label (if available)
            steps_dict = {}
            env_ref = getattr(ag, "ma_problem", None)
            if env_ref and getattr(env_ref, "agent_steps", None) is not None:
                steps_dict = env_ref.agent_steps
            elif getattr(self, "agent_steps_view", None) is not None:
                steps_dict = getattr(self, "agent_steps_view")
            steps_dict = steps_dict or {}
            if ag.name in steps_dict:
                steps_label = f"steps: {steps_dict.get(ag.name, 0)}"
                steps_txt = self.font.render(steps_label, True, (40, 40, 40))
                self.screen.blit(steps_txt, (panel_x, y))
                y += steps_txt.get_height() + 2

            # State label
            try:
                idx = rm.get_state_index(current_state)
                state_label = f"state: {current_state} (q={idx})"
            except Exception:
                state_label = f"state: {current_state}"
            state_txt = self.font.render(state_label, True, (30, 30, 30))
            self.screen.blit(state_txt, (panel_x, y))
            y += state_txt.get_height() + 2

            # Event label (detected on current obs)
            event_val = None
            try:
                event_val = rm.event_detector.detect_event(obs.get(ag.name, {}))
            except Exception:
                event_val = None
            event_label = (
                f"event: {event_val}" if event_val is not None else "event: None"
            )
            event_txt = self.font.render(event_label, True, (60, 60, 60))
            self.screen.blit(event_txt, (panel_x, y))
            y += event_txt.get_height() + 6

            # Draw bullet timeline for RM states
            n_states = len(states_ordered)
            if n_states > 1:
                available = self.panel_width - 40
                spacing = max(26, min(70, available / (n_states - 1)))
            else:
                spacing = 0
            radius = int(max(8, min(16, spacing / 2 if spacing else 10)))

            timeline_width = (n_states - 1) * spacing if n_states > 1 else 0
            x_start = panel_left + (self.panel_width - timeline_width) / 2
            for idx, state_name in enumerate(states_ordered):
                cx = x_start + idx * spacing
                cy = y + radius
                is_final = state_name == final_state
                base_color = (0, 120, 0) if is_final else (0, 0, 0)

                # Base circle (white) and connectors
                pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), radius)
                pygame.draw.circle(self.screen, (50, 50, 50), (cx, cy), radius, 1)

                # Highlight the current state with a bright fill/halo
                if state_name == current_state:
                    highlight = (180, 240, 180)  # light green
                    pygame.draw.circle(self.screen, highlight, (cx, cy), radius + 3)
                    pygame.draw.circle(self.screen, highlight, (cx, cy), radius)
                    pygame.draw.circle(self.screen, base_color, (cx, cy), radius, 1)
                else:
                    pygame.draw.circle(self.screen, base_color, (cx, cy), radius, 1)

                # Label inside the circle
                label = f"s{idx}"
                # Scale font to circle size for readability
                font_size = max(10, min(18, int(radius * 1.6)))
                label_font = pygame.font.SysFont("Arial", font_size)
                txt = label_font.render(label, True, (0, 0, 0))
                txt_rect = txt.get_rect(center=(cx, cy))
                self.screen.blit(txt, txt_rect)
                # Draw connector line to next state
                if idx < len(states_ordered) - 1:
                    next_cx = x_start + (idx + 1) * spacing
                    pygame.draw.line(
                        self.screen,
                        (100, 100, 100),
                        (cx + radius, cy),
                        (next_cx - radius, cy),
                        2,
                    )
            y += radius * 2 + 10

        if not has_rm:
            msg = self.font.render("No RM attached", True, (90, 90, 90))
            self.screen.blit(msg, (panel_x, y))

    """def save_episode(self, episode):

        if self.frames:
            video_path = f"episodes/episode_{episode}.avi"
            height, width, layers = self.frames[0].shape
            video = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"DIVX"), 2, (width, height)
            )

            for frame in self.frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            cv2.destroyAllWindows()
            video.release()
            self.frames = []  # Clear the frame list"""

    import imageio  # Ensure imageio is imported

    def save_episode(self, episode, wandb=None):
        """
        Save the episode as a video and a GIF. If a WandB object is provided,
        log the files to WandB instead of just saving them locally.

        :param episode: The episode number.
        :param wandb: A WandB object for logging (optional).
        """
        # Create the "episodes" directory if it doesn't exist (for temporary local saving)
        episodes_dir = "episodes"
        if not os.path.exists(episodes_dir):
            os.makedirs(episodes_dir, exist_ok=True)
        # if episode == 0:
        #    self.save_first_frame("first_frame.png")

        if self.frames:
            # Save as avi
            video_path = f"episodes/episode_{episode}.avi"
            height, width, layers = self.frames[0].shape
            video = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"DIVX"), 2, (width, height)
            )

            for frame in self.frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            video.release()

            # Save as GIF
            gif_path = f"{episodes_dir}/episode_{episode}.gif"
            try:
                imageio.mimsave(gif_path, self.frames, fps=2, loop=0)
            except Exception as e:
                print(f"Error while creating GIF: {e}")
                return  # Exit early to avoid downstream failures

            if not os.path.isfile(gif_path):
                print(f"Error: GIF {gif_path} was not created.")
                return

            # If WandB is provided, log the files to WandB
            if wandb:
                wandb.log(
                    {
                        # f"Episode {episode} Video": wandb.Video(video_path, format="mp4"),
                        f"Episode {episode} GIF": wandb.Image(gif_path)
                    }
                )

                # Delete local files after uploading to WandB
                # os.remove(video_path)
                # os.remove(gif_path)
                print(f"Episode {episode} saved to WandB.")
            else:
                print(f"Files saved locally: {video_path}, {gif_path}")

            # Clear the frames after saving
            self.frames = []

    # Additional rendering helpers

    def save_first_frame(self, filename="first_frame.png"):
        """Save the first rendered frame as a PNG."""
        if self.frames:
            first_frame = self.frames[0]
            # Convert the frame into a Pygame surface
            surface = pygame.surfarray.make_surface(first_frame.transpose([1, 0, 2]))
            # Save the surface as a PNG image
            pygame.image.save(surface, filename)
            print(f"First frame saved as {filename}")
        else:
            print("No frame available to save.")

    def simulate_agents(self, agent_paths):
        """Simulate agent trajectories and render each timestep before exporting."""
        max_steps = max(
            len(path) for path in agent_paths.values()
        )  # Find the maximum number of steps any agent takes

        for step in range(max_steps):
            # Create a dictionary to store the current state of each agent
            obs = {}
            for agent, path in agent_paths.items():
                if step < len(path):
                    # Unpack position and convert indices (adjust if the coordinate system differs)
                    x, y, _ = path[step]
                    obs[agent] = {"pos_x": x, "pos_y": y}
                else:
                    # Use the last position if the path has ended for this agent
                    x, y, _ = path[-1]
                    obs[agent] = {"pos_x": x, "pos_y": y}

            # Render the current state of the environment
            self.render(step, obs)

            # Add logic to handle game logic here if necessary (e.g., checking for collisions)

        # Save the episode to a video or gif after the simulation
        self.save_episode("final_simulation")
