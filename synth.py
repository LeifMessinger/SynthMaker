import numpy as np

# Implement the Synth and Wave classes with optimizations
class Synth:
    def __init__(self, sampleFrequency=48000):
        self.sampleFrequency = sampleFrequency
    
    def __call__(self, t):
        """Sample the synthesizer at a specific point in time t (in seconds)"""
        raise NotImplementedError("Subclasses must implement __call__")
    
    def sample(self, duration, startTime=0):
        """Create an array of samples for the given duration starting at startTime"""
        num_samples = int(duration * self.sampleFrequency)
        times = np.linspace(startTime, startTime + duration, num_samples, endpoint=False)
        return np.array([self(t) for t in times])

class Wave(Synth):
    def __init__(self, frequency=440, phase=0, volume=1, sampleFrequency=48000):
        super().__init__(sampleFrequency)
        self.frequency = frequency
        self.phase = phase
        self.volume = volume
    
    def __call__(self, t):
        """Generate a sine wave sample at time t with the specified phase and volume"""
        phase_radians = self.phase * 2 * np.pi
        return self.volume * np.sin(2 * np.pi * self.frequency * t + phase_radians)
    
    def sample(self, duration, startTime=0):
        """Optimized version that generates the entire array at once using numpy"""
        num_samples = int(duration * self.sampleFrequency)
        times = np.linspace(startTime, startTime + duration, num_samples, endpoint=False)
        phase_radians = self.phase * 2 * np.pi
        return self.volume * np.sin(2 * np.pi * self.frequency * times + phase_radians)