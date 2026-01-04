import { useState } from 'react'
import { motion } from 'framer-motion'

const SAMPLE_POLICY = `name: sample_policy
domain: natural
transforms:
  - name: HorizontalFlip
    probability: 0.5
  - name: Rotate
    probability: 0.3
    parameters:
      limit: 15
  - name: RandomBrightnessContrast
    probability: 0.4
  - name: GaussNoise
    probability: 0.2`

function Ablation() {
    const [policyYaml, setPolicyYaml] = useState(SAMPLE_POLICY)
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)

    const runAblation = async () => {
        setLoading(true)
        try {
            const res = await fetch('/api/ablation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ policy_yaml: policyYaml, mock: true }),
            })
            if (res.ok) {
                setResult(await res.json())
            }
        } catch (err) {
            console.error(err)
        }
        setLoading(false)
    }

    return (
        <div className="max-w-6xl mx-auto">
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8"
            >
                <h1 className="text-2xl font-bold tracking-widest mb-2">
                    ABLATION :: ANALYZER
                </h1>
                <p className="text-sm opacity-70">
                    Measure the contribution of each transform via leave-one-out analysis
                </p>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Policy Input */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="card"
                >
                    <h2 className="text-sm font-bold tracking-widest mb-4">POLICY_YAML</h2>
                    <textarea
                        value={policyYaml}
                        onChange={(e) => setPolicyYaml(e.target.value)}
                        className="w-full h-64 bg-black text-white font-mono text-xs p-4 mb-4"
                        placeholder="Paste policy YAML here..."
                    />
                    <button
                        onClick={runAblation}
                        disabled={loading || !policyYaml.trim()}
                        className="pixel-btn w-full"
                    >
                        {loading ? 'ANALYZING...' : 'RUN ABLATION'}
                    </button>
                </motion.div>

                {/* Results */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="card"
                >
                    <h2 className="text-sm font-bold tracking-widest mb-4">CONTRIBUTIONS</h2>
                    {result ? (
                        <div className="space-y-3">
                            <div className="text-center mb-4">
                                <span className="text-[10px] tracking-widest">BASELINE_SCORE</span>
                                <p className="text-3xl font-bold">{(result.baseline_score * 100).toFixed(1)}%</p>
                            </div>
                            {result.contributions.map((c) => (
                                <div key={c.name} className="border-2 border-black p-3">
                                    <div className="flex justify-between mb-2">
                                        <span className="font-bold text-sm">#{c.rank} {c.name}</span>
                                        <span className={`text-sm ${c.is_helpful ? 'text-green-600' : 'text-red-600'}`}>
                                            {c.contribution >= 0 ? '+' : ''}{(c.contribution * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="flex justify-between text-[10px] opacity-70">
                                        <span>Impact: {c.impact_label}</span>
                                        <span>Ablated: {(c.ablated_score * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                            ))}
                            <div className="text-[10px] mt-4 border-t border-black/20 pt-4">
                                {result.recommended_keeps?.length > 0 && (
                                    <p className="text-green-600">✓ Most helpful: {result.recommended_keeps.join(', ')}</p>
                                )}
                                {result.recommended_removes?.length > 0 && (
                                    <p className="text-yellow-600 mt-1">⚠ Consider removing: {result.recommended_removes.join(', ')}</p>
                                )}
                            </div>
                        </div>
                    ) : (
                        <p className="text-sm opacity-50 text-center py-8">
                            Paste a policy YAML and run ablation to see results
                        </p>
                    )}
                </motion.div>
            </div>
        </div>
    )
}

export default Ablation
