"""
AgentEvolver Launcher API
FastAPI endpoint for managing agent evolution
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import uuid
from datetime import datetime

from .agent_evolver import AgentEvolver, EvolutionConfig, AgentPerformance


app = FastAPI(
    title="AgentEvolver API",
    description="Self-evolving agent system for Ajatuskumppani",
    version="1.0.0"
)

# Store active evolution sessions
evolution_sessions: Dict[str, Dict[str, Any]] = {}


class StartEvolutionRequest(BaseModel):
    """Request to start evolution"""
    model_name: str = "mistral-7b"
    evolution_cycles: int = 10
    agents_per_cycle: int = 5
    use_appworld: bool = True
    use_reme: bool = True


class EvolutionStatus(BaseModel):
    """Evolution session status"""
    session_id: str
    status: str  # running, completed, failed
    current_generation: int
    total_generations: int
    best_success_rate: float
    avg_success_rate: float
    started_at: str
    completed_at: Optional[str] = None


@app.post("/evolution/start", response_model=Dict[str, str])
async def start_evolution(
    request: StartEvolutionRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new evolution session
    
    Returns:
        session_id: Unique identifier for the evolution session
    """
    session_id = str(uuid.uuid4())
    
    # Create configuration
    config = EvolutionConfig(
        model_name=request.model_name,
        evolution_cycles=request.evolution_cycles,
        agents_per_cycle=request.agents_per_cycle,
        use_appworld=request.use_appworld,
        use_reme=request.use_reme
    )
    
    # Initialize session
    evolution_sessions[session_id] = {
        "status": "initializing",
        "config": config,
        "evolver": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    # Start evolution in background
    background_tasks.add_task(run_evolution, session_id, config)
    
    return {
        "session_id": session_id,
        "message": "Evolution started"
    }


@app.get("/evolution/{session_id}/status", response_model=EvolutionStatus)
async def get_evolution_status(session_id: str):
    """
    Get status of an evolution session
    
    Args:
        session_id: Evolution session ID
    
    Returns:
        Current status and statistics
    """
    if session_id not in evolution_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = evolution_sessions[session_id]
    evolver = session.get("evolver")
    
    if evolver is None:
        return EvolutionStatus(
            session_id=session_id,
            status=session["status"],
            current_generation=0,
            total_generations=session["config"].evolution_cycles,
            best_success_rate=0.0,
            avg_success_rate=0.0,
            started_at=session["started_at"],
            completed_at=session.get("completed_at")
        )
    
    stats = evolver.get_statistics()
    
    return EvolutionStatus(
        session_id=session_id,
        status=session["status"],
        current_generation=stats.get("current_generation", 0),
        total_generations=session["config"].evolution_cycles,
        best_success_rate=stats.get("best_success_rate", 0.0),
        avg_success_rate=stats.get("current_avg_success_rate", 0.0),
        started_at=session["started_at"],
        completed_at=session.get("completed_at")
    )


@app.get("/evolution/{session_id}/best-agent")
async def get_best_agent(session_id: str):
    """
    Get the best agent from an evolution session
    
    Args:
        session_id: Evolution session ID
    
    Returns:
        Best agent configuration and performance
    """
    if session_id not in evolution_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = evolution_sessions[session_id]
    evolver = session.get("evolver")
    
    if evolver is None:
        raise HTTPException(status_code=400, detail="Evolution not started")
    
    best_agent = evolver._get_best_agent()
    
    if best_agent is None:
        raise HTTPException(status_code=404, detail="No agents evaluated yet")
    
    return {
        "agent": best_agent["agent"],
        "performance": best_agent["performance"].to_dict()
    }


@app.get("/evolution/{session_id}/history")
async def get_evolution_history(session_id: str):
    """
    Get evolution history for a session
    
    Args:
        session_id: Evolution session ID
    
    Returns:
        Performance history for all generations
    """
    if session_id not in evolution_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = evolution_sessions[session_id]
    evolver = session.get("evolver")
    
    if evolver is None:
        raise HTTPException(status_code=400, detail="Evolution not started")
    
    return {
        "history": [p.to_dict() for p in evolver.performance_history]
    }


@app.delete("/evolution/{session_id}")
async def stop_evolution(session_id: str):
    """
    Stop and delete an evolution session
    
    Args:
        session_id: Evolution session ID
    
    Returns:
        Confirmation message
    """
    if session_id not in evolution_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # TODO: Implement graceful shutdown of evolution
    
    del evolution_sessions[session_id]
    
    return {"message": "Evolution session stopped"}


@app.get("/evolution/sessions")
async def list_sessions():
    """
    List all evolution sessions
    
    Returns:
        List of session IDs and their status
    """
    return {
        "sessions": [
            {
                "session_id": session_id,
                "status": session["status"],
                "started_at": session["started_at"]
            }
            for session_id, session in evolution_sessions.items()
        ]
    }


async def run_evolution(session_id: str, config: EvolutionConfig):
    """
    Run evolution in background
    
    Args:
        session_id: Session ID
        config: Evolution configuration
    """
    try:
        # Update status
        evolution_sessions[session_id]["status"] = "running"
        
        # Create evolver
        evolver = AgentEvolver(config)
        evolution_sessions[session_id]["evolver"] = evolver
        
        # Initialize
        await evolver.initialize()
        
        # Run evolution
        await evolver.evolve()
        
        # Update status
        evolution_sessions[session_id]["status"] = "completed"
        evolution_sessions[session_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        evolution_sessions[session_id]["status"] = "failed"
        evolution_sessions[session_id]["error"] = str(e)
        evolution_sessions[session_id]["completed_at"] = datetime.now().isoformat()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "AgentEvolver"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

