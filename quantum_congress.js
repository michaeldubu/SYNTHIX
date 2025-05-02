import React, { useState, useEffect } from 'react';
import { Globe, Vote, Landmark, Scale, Users, Timer, AlertTriangle, CheckCircle, XCircle, Library } from 'lucide-react';

const QuantumCongress = () => {
  const [activePolicies, setActivePolicies] = useState([
    {
      id: 1,
      title: "Universal Resource Exchange",
      proposedBy: "Neo-Quantum Republic",
      support: 65,
      opposition: 35,
      timeRemaining: "6:24:00",
      impact: {
        economy: 0.8,
        diplomacy: 0.6,
        technology: 0.4
      },
      votes: {
        for: ["Neo-Quantum Republic", "Synthetic Collective"],
        against: ["Crystal Dynasty"],
        abstain: ["Quantum Ascendancy"]
      }
    },
    {
      id: 2,
      title: "Quantum Technology Sanctions",
      proposedBy: "Synthetic Collective",
      support: 45,
      opposition: 55,
      timeRemaining: "12:48:00",
      emergency: true,
      impact: {
        economy: -0.4,
        diplomacy: -0.2,
        technology: 0.8
      },
      votes: {
        for: ["Synthetic Collective", "Neo-Quantum Republic"],
        against: ["Crystal Dynasty", "Quantum Ascendancy"],
        abstain: []
      }
    }
  ]);

  const [congressStats, setCongressStats] = useState({
    totalMembers: 8,
    activeMembers: 6,
    policiesEnacted: 24,
    congressPower: 82
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-purple-900 p-6">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Congress Header */}
        <header className="bg-black/30 rounded-xl p-6 backdrop-blur-md border border-purple-500/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <Landmark className="w-10 h-10 text-purple-400" />
                <Globe className="w-5 h-5 text-blue-400 absolute -top-2 -right-2" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Quantum Congress™</h1>
                <p className="text-purple-400">Universal Policy Management</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-6">
              {Object.entries(congressStats).map(([key, value]) => (
                <div key={key} className="text-center">
                  <div className="text-2xl font-bold text-white">{value}</div>
                  <div className="text-sm text-purple-400 capitalize">{key.replace(/([A-Z])/g, ' $1')}</div>
                </div>
              ))}
            </div>
          </div>
        </header>

        {/* Active Policies */}
        <div className="space-y-6">
          {activePolicies.map(policy => (
            <div key={policy.id} className="bg-black/30 rounded-xl p-6 backdrop-blur-md border border-purple-500/20">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                  {policy.emergency ? (
                    <AlertTriangle className="w-6 h-6 text-red-400" />
                  ) : (
                    <Library className="w-6 h-6 text-purple-400" />
                  )}
                  <div>
                    <h3 className="text-lg font-bold text-white">{policy.title}</h3>
                    <p className="text-purple-400">Proposed by {policy.proposedBy}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <Timer className="w-5 h-5 text-blue-400" />
                  <span className="text-blue-400">{policy.timeRemaining}</span>
                </div>
              </div>

              {/* Voting Progress */}
              <div className="space-y-6">
                <div className="relative h-3 bg-black/50 rounded-full overflow-hidden">
                  <div 
                    className="absolute left-0 h-full bg-green-500"
                    style={{ width: `${policy.support}%` }}
                  />
                  <div 
                    className="absolute right-0 h-full bg-red-500"
                    style={{ width: `${policy.opposition}%` }}
                  />
                </div>

                {/* Impact Assessment */}
                <div className="grid grid-cols-3 gap-4">
                  {Object.entries(policy.impact).map(([area, value]) => (
                    <div key={area} className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-purple-400 capitalize">{area}</span>
                        <span className={`text-${value > 0 ? 'green' : 'red'}-400`}>
                          {value > 0 ? '+' : ''}{(value * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-2 bg-black/50 rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${value > 0 ? 'bg-green-400' : 'bg-red-400'}`}
                          style={{ width: `${Math.abs(value * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Voting Details */}
                <div className="grid grid-cols-3 gap-4">
                  {Object.entries(policy.votes).map(([voteType, civilizations]) => (
                    <div key={voteType} className="space-y-2">
                      <div className="flex items-center space-x-2">
                        {voteType === 'for' && <CheckCircle className="w-4 h-4 text-green-400" />}
                        {voteType === 'against' && <XCircle className="w-4 h-4 text-red-400" />}
                        {voteType === 'abstain' && <Scale className="w-4 h-4 text-yellow-400" />}
                        <span className="text-purple-400 capitalize">{voteType}</span>
                      </div>
                      <div className="space-y-1">
                        {civilizations.map(civ => (
                          <div key={civ} className="text-sm text-white bg-black/30 rounded px-2 py-1">
                            {civ}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Policy Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <button className="bg-purple-500 hover:bg-purple-600 text-white rounded-xl p-4 flex items-center justify-center space-x-2">
            <Vote className="w-5 h-5" />
            <span>Propose New Policy</span>
          </button>
          <button className="bg-blue-500 hover:bg-blue-600 text-white rounded-xl p-4 flex items-center justify-center space-x-2">
            <Users className="w-5 h-5" />
            <span>Call Emergency Session</span>
          </button>
          <button className="bg-green-500 hover:bg-green-600 text-white rounded-xl p-4 flex items-center justify-center space-x-2">
            <Scale className="w-5 h-5" />
            <span>Review Past Policies</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default QuantumCongress;
