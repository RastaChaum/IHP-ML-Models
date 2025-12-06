"""Unit tests for MemoryReplayBuffer."""

from datetime import datetime

import pytest
from domain.value_objects import (
    EntityState,
    HeatingActionType,
    RLAction,
    RLExperience,
    RLObservation,
)
from infrastructure.adapters import MemoryReplayBuffer


@pytest.fixture
def sample_observation():
    """Create a sample RLObservation for testing."""
    return RLObservation(
        indoor_temp=20.0,
        indoor_temp_entity=EntityState(entity_id="sensor.indoor_temp", last_changed_minutes=0.0),
        outdoor_temp=10.0,
        outdoor_temp_entity=EntityState(entity_id="sensor.outdoor_temp", last_changed_minutes=0.0),
        indoor_humidity=50.0,
        indoor_humidity_entity=EntityState(entity_id="sensor.humidity", last_changed_minutes=0.0),
        timestamp=datetime.now(),
        target_temp=22.0,
        target_temp_entity=EntityState(entity_id="sensor.target_temp", last_changed_minutes=0.0),
        time_until_target_minutes=0,
        current_target_achieved_percentage=80.0,
        is_heating_on=True,
        heating_output_percent=None,
        heating_output_entity=None,
        energy_consumption_recent_kwh=None,
        energy_consumption_entity=None,
        time_heating_on_recent_seconds=None,
        time_heating_on_entity=None,
        indoor_temp_change_15min=0.5,
        outdoor_temp_change_15min=-0.2,
        day_of_week=1,
        hour_of_day=10,
        outdoor_temp_forecast_1h=9.0,
        outdoor_temp_forecast_3h=8.0,
        window_or_door_open=False,
        window_or_door_entity=None,
        device_id="zone_1",
    )


@pytest.fixture
def sample_action():
    """Create a sample RLAction for testing."""
    return RLAction(
        action_type=HeatingActionType.TURN_ON,
        value=22.0,
        decision_timestamp=datetime.now(),
        confidence_score=0.9,
    )


@pytest.fixture
def sample_experience(sample_observation, sample_action):
    """Create a sample RLExperience for testing."""
    next_obs = RLObservation(
        indoor_temp=20.5,
        indoor_temp_entity=EntityState(entity_id="sensor.indoor_temp", last_changed_minutes=5.0),
        outdoor_temp=10.0,
        outdoor_temp_entity=EntityState(entity_id="sensor.outdoor_temp", last_changed_minutes=5.0),
        indoor_humidity=50.0,
        indoor_humidity_entity=EntityState(entity_id="sensor.humidity", last_changed_minutes=5.0),
        timestamp=datetime.now(),
        target_temp=22.0,
        target_temp_entity=EntityState(entity_id="sensor.target_temp", last_changed_minutes=0.0),
        time_until_target_minutes=0,
        current_target_achieved_percentage=90.0,
        is_heating_on=True,
        heating_output_percent=None,
        heating_output_entity=None,
        energy_consumption_recent_kwh=None,
        energy_consumption_entity=None,
        time_heating_on_recent_seconds=None,
        time_heating_on_entity=None,
        indoor_temp_change_15min=0.5,
        outdoor_temp_change_15min=-0.2,
        day_of_week=1,
        hour_of_day=10,
        outdoor_temp_forecast_1h=9.0,
        outdoor_temp_forecast_3h=8.0,
        window_or_door_open=False,
        window_or_door_entity=None,
        device_id="zone_1",
    )
    
    return RLExperience(
        state=sample_observation,
        action=sample_action,
        reward=0.5,
        next_state=next_obs,
        done=False,
    )


@pytest.mark.asyncio
async def test_buffer_initialization():
    """Test buffer initialization."""
    buffer = MemoryReplayBuffer(max_capacity=100)
    
    assert await buffer.size() == 0
    assert await buffer.is_ready(min_size=10) is False


@pytest.mark.asyncio
async def test_buffer_add_single_experience(sample_experience):
    """Test adding a single experience to the buffer."""
    buffer = MemoryReplayBuffer(max_capacity=100)
    
    await buffer.add(sample_experience)
    
    assert await buffer.size() == 1
    assert await buffer.size(device_id="zone_1") == 1


@pytest.mark.asyncio
async def test_buffer_add_batch_experiences(sample_experience):
    """Test adding a batch of experiences."""
    buffer = MemoryReplayBuffer(max_capacity=100)
    
    experiences = tuple([sample_experience] * 10)
    await buffer.add_batch(experiences)
    
    assert await buffer.size() == 10


@pytest.mark.asyncio
async def test_buffer_sample(sample_experience):
    """Test sampling from the buffer."""
    buffer = MemoryReplayBuffer(max_capacity=100)
    
    # Add experiences
    experiences = tuple([sample_experience] * 20)
    await buffer.add_batch(experiences)
    
    # Sample a batch
    sampled = await buffer.sample(batch_size=5)
    
    assert len(sampled) == 5
    assert all(isinstance(exp, RLExperience) for exp in sampled)


@pytest.mark.asyncio
async def test_buffer_sample_empty_raises_error():
    """Test sampling from empty buffer raises error."""
    buffer = MemoryReplayBuffer(max_capacity=100)
    
    with pytest.raises(ValueError, match="Cannot sample from empty buffer"):
        await buffer.sample(batch_size=5)


@pytest.mark.asyncio
async def test_buffer_sample_insufficient_data_raises_error(sample_experience):
    """Test sampling more than available raises error."""
    buffer = MemoryReplayBuffer(max_capacity=100)
    
    await buffer.add(sample_experience)
    
    with pytest.raises(ValueError, match="Buffer has only 1 experiences"):
        await buffer.sample(batch_size=5)


@pytest.mark.asyncio
async def test_buffer_clear_all(sample_experience):
    """Test clearing all experiences."""
    buffer = MemoryReplayBuffer(max_capacity=100)
    
    # Add experiences
    experiences = tuple([sample_experience] * 10)
    await buffer.add_batch(experiences)
    
    assert await buffer.size() == 10
    
    # Clear all
    await buffer.clear()
    
    assert await buffer.size() == 0


@pytest.mark.asyncio
async def test_buffer_clear_device_specific(sample_experience, sample_observation, sample_action):
    """Test clearing experiences for a specific device."""
    buffer = MemoryReplayBuffer(max_capacity=100)
    
    # Add experiences for zone_1
    await buffer.add_batch(tuple([sample_experience] * 5))
    
    # Create experience for zone_2
    obs_zone2 = RLObservation(
        **{**sample_observation.__dict__, "device_id": "zone_2"}
    )
    next_obs_zone2 = RLObservation(
        **{**sample_observation.__dict__, "indoor_temp": 21.0, "device_id": "zone_2"}
    )
    exp_zone2 = RLExperience(
        state=obs_zone2,
        action=sample_action,
        reward=0.3,
        next_state=next_obs_zone2,
        done=False,
    )
    await buffer.add_batch(tuple([exp_zone2] * 3))
    
    assert await buffer.size() == 8
    assert await buffer.size(device_id="zone_1") == 5
    assert await buffer.size(device_id="zone_2") == 3
    
    # Clear zone_1
    await buffer.clear(device_id="zone_1")
    
    assert await buffer.size() == 3
    assert await buffer.size(device_id="zone_1") == 0
    assert await buffer.size(device_id="zone_2") == 3


@pytest.mark.asyncio
async def test_buffer_is_ready(sample_experience):
    """Test checking if buffer is ready for training."""
    buffer = MemoryReplayBuffer(max_capacity=100)
    
    assert await buffer.is_ready(min_size=10) is False
    
    # Add experiences
    experiences = tuple([sample_experience] * 15)
    await buffer.add_batch(experiences)
    
    assert await buffer.is_ready(min_size=10) is True
    assert await buffer.is_ready(min_size=20) is False


@pytest.mark.asyncio
async def test_buffer_get_all(sample_experience):
    """Test getting all experiences from the buffer."""
    buffer = MemoryReplayBuffer(max_capacity=100)
    
    # Add experiences
    experiences = tuple([sample_experience] * 10)
    await buffer.add_batch(experiences)
    
    all_exps = await buffer.get_all()
    
    assert len(all_exps) == 10
    assert all(isinstance(exp, RLExperience) for exp in all_exps)


@pytest.mark.asyncio
async def test_buffer_max_capacity(sample_experience):
    """Test buffer respects maximum capacity."""
    buffer = MemoryReplayBuffer(max_capacity=10)
    
    # Add more experiences than capacity
    experiences = tuple([sample_experience] * 15)
    await buffer.add_batch(experiences)
    
    # Buffer should only contain the last 10 experiences (FIFO)
    assert await buffer.size() == 10
