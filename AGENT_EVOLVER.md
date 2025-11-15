# AgentEvolver Integration

Ajatuskumppani integrates **AgentEvolver** - a self-evolving agent system that automatically improves AI agents through evolutionary algorithms.

## Overview

AgentEvolver uses evolutionary computation to optimize agent performance:

1. **Population Generation**: Create diverse agents with different hyperparameters
2. **Evaluation**: Test agents in AppWorld environment
3. **Selection**: Choose best performing agents
4. **Evolution**: Create new generation through crossover and mutation
5. **Fine-tuning**: Train models on successful trajectories

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentEvolver System                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Population │───▶│  Evaluation  │───▶│  Selection   │  │
│  │  Generation  │    │  (AppWorld)  │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         ▲                                        │           │
│         │                                        ▼           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Fine-tuning │◀───│   Crossover  │◀───│   Mutation   │  │
│  │    (LoRA)    │    │  & Mutation  │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Install Dependencies

```bash
cd ajatus-agents

# Install Python dependencies
pip install -r requirements.txt

# Install AgentEvolver
pip install agent-evolver
```

### 2. Setup AppWorld Environment

```bash
cd env_service/environments/appworld
bash setup.sh
```

### 3. Install ReMe (Optional)

```bash
bash external/reme/install_reme.sh
```

### 4. Activate Environment

```bash
conda activate agentevolver
```

## Usage

### Option 1: Python API

```python
from src.evolver.agent_evolver import AgentEvolver, EvolutionConfig

# Create configuration
config = EvolutionConfig(
    model_name="mistral-7b",
    evolution_cycles=10,
    agents_per_cycle=5,
    use_appworld=True,
    use_reme=True
)

# Initialize evolver
evolver = AgentEvolver(config)
await evolver.initialize()

# Run evolution
best_agent = await evolver.evolve()

print(f"Best Success Rate: {best_agent['performance'].success_rate:.2%}")
```

### Option 2: REST API

Start the API server:

```bash
python src/evolver/launcher.py
```

Start evolution:

```bash
curl -X POST http://localhost:8001/evolution/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "mistral-7b",
    "evolution_cycles": 10,
    "agents_per_cycle": 5,
    "use_appworld": true,
    "use_reme": true
  }'
```

Check status:

```bash
curl http://localhost:8001/evolution/{session_id}/status
```

Get best agent:

```bash
curl http://localhost:8001/evolution/{session_id}/best-agent
```

### Option 3: Command Line

Minimal example (without ReMe):

```bash
python launcher.py --conf examples/basic.yaml --with-appworld
```

Full example (with ReMe):

```bash
python launcher.py --conf examples/overall.yaml --with-appworld --with-reme
```

## Configuration

### Evolution Settings

```yaml
# Evolution configuration
evolution:
  cycles: 10                    # Number of evolution cycles
  agents_per_cycle: 5           # Population size per generation
  selection_strategy: tournament # Selection method

# Model settings
model:
  name: mistral-7b
  base_path: ./models/mistral-7b

# Training settings
training:
  batch_size: 4
  learning_rate: 1e-5
  num_epochs: 3

# Environment settings
environment:
  use_appworld: true
  use_reme: true

# Performance thresholds
thresholds:
  min_success_rate: 0.7
  min_efficiency_score: 0.6
```

## AppWorld Integration

AppWorld provides realistic task environments for agent evaluation:

- **Task Types**: Email, calendar, contacts, notes, web browsing
- **Metrics**: Success rate, efficiency, response time
- **Scenarios**: 100+ realistic user scenarios

## ReMe Integration

ReMe (Retrieval-Enhanced Memory) provides:

- **Questioning**: Ask clarifying questions
- **Navigating**: Explore information space
- **Attributing**: Track information sources

## Evolution Strategies

### Selection Methods

1. **Tournament Selection** (default)
   - Select best from random subsets
   - Maintains diversity

2. **Elitist Selection**
   - Always keep top performers
   - Fast convergence

3. **Roulette Selection**
   - Probability based on fitness
   - Balanced exploration

### Mutation Operations

- **Temperature**: ±0.1
- **Top-p**: ±0.05
- **Max Tokens**: ±100
- **Reasoning Depth**: ±1
- **Exploration Rate**: ±0.05

### Crossover

- Uniform crossover
- Each gene randomly selected from either parent

## Performance Metrics

### Agent Performance

- **Success Rate**: Percentage of successfully completed tasks
- **Efficiency Score**: Resource usage and time efficiency
- **Response Time**: Average time per task
- **Task Completion**: Total/successful/failed tasks

### Evolution Statistics

- **Best Success Rate**: Highest across all generations
- **Average Success Rate**: Current generation average
- **Generation Progress**: Current/total generations
- **Best Agent ID**: Identifier of top performer

## Fine-tuning

Best agents are fine-tuned every 3 generations:

1. **Collect Trajectories**: Gather successful task completions
2. **Create Dataset**: Format for training
3. **Fine-tune Model**: Use LoRA or full fine-tuning
4. **Save Checkpoint**: Store fine-tuned model

## API Endpoints

### Start Evolution

```
POST /evolution/start
```

### Get Status

```
GET /evolution/{session_id}/status
```

### Get Best Agent

```
GET /evolution/{session_id}/best-agent
```

### Get History

```
GET /evolution/{session_id}/history
```

### Stop Evolution

```
DELETE /evolution/{session_id}
```

### List Sessions

```
GET /evolution/sessions
```

## Citation

If you use AgentEvolver in your research, please cite:

```bibtex
@misc{AgentEvolver2025,
  title         = {AgentEvolver: Towards Efficient Self-Evolving Agent System},
  author        = {Yunpeng Zhai and Shuchang Tao and Cheng Chen and Anni Zou and Ziqian Chen and Qingxu Fu and Shinji Mai and Li Yu and Jiaji Deng and Zouying Cao and Zhaoyang Liu and Bolin Ding and Jingren Zhou},
  year          = {2025},
  eprint        = {2511.10395},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2511.10395}
}
```

## License

AgentEvolver integration follows the same license as Ajatuskumppani (AGPL 3.0).

## Support

- GitHub Issues: https://github.com/Pinnacore-AI/ajatus-agents/issues
- Discord: https://discord.gg/z53hngJHd
- Email: ajatuskumppani@pinnacore.ai

