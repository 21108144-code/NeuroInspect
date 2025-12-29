import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { analyzeRootCauses } from '@/lib/api'
import { formatPercent, formatDateTime } from '@/lib/utils'
import type { RootCauseResponse, ClusterInfo } from '@/lib/types'
import {
    GitBranch,
    Play,
    RefreshCw,
    AlertCircle,
    Lightbulb,
    Target,
} from 'lucide-react'

export default function RootCause() {
    const [result, setResult] = useState<RootCauseResponse | null>(null)
    const [loading, setLoading] = useState(false)
    const [timeRange, setTimeRange] = useState(24)
    const [minClusterSize, setMinClusterSize] = useState(5)

    const handleAnalyze = async () => {
        setLoading(true)
        try {
            const data = await analyzeRootCauses({
                timeRangeHours: timeRange,
                minClusterSize,
                minSamples: 3,
            })
            setResult(data)
        } catch (error) {
            console.error('Root cause analysis failed:', error)
        } finally {
            setLoading(false)
        }
    }

    return (
        <main className="p-6 space-y-6">
            {/* Controls */}
            <Card className="glass-card">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <GitBranch className="w-5 h-5" />
                        Analysis Configuration
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="space-y-2">
                            <label className="text-sm font-medium">Time Range (hours)</label>
                            <div className="flex items-center gap-4">
                                <Slider
                                    value={[timeRange]}
                                    onValueChange={(v) => setTimeRange(v[0])}
                                    min={1}
                                    max={168}
                                    step={1}
                                    className="flex-1"
                                />
                                <span className="text-sm font-mono w-12">{timeRange}h</span>
                            </div>
                        </div>
                        <div className="space-y-2">
                            <label className="text-sm font-medium">Min Cluster Size</label>
                            <div className="flex items-center gap-4">
                                <Slider
                                    value={[minClusterSize]}
                                    onValueChange={(v) => setMinClusterSize(v[0])}
                                    min={2}
                                    max={20}
                                    step={1}
                                    className="flex-1"
                                />
                                <span className="text-sm font-mono w-12">{minClusterSize}</span>
                            </div>
                        </div>
                        <div className="flex items-end">
                            <Button onClick={handleAnalyze} disabled={loading} className="w-full">
                                {loading ? (
                                    <>
                                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                                        Analyzing...
                                    </>
                                ) : (
                                    <>
                                        <Play className="w-4 h-4 mr-2" />
                                        Run Analysis
                                    </>
                                )}
                            </Button>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Results Summary */}
            {result && (
                <>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <Card className="glass-card p-4">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-full bg-primary/20">
                                    <Target className="w-5 h-5 text-primary" />
                                </div>
                                <div>
                                    <p className="text-sm text-muted-foreground">Defects Analyzed</p>
                                    <p className="text-2xl font-bold">{result.total_defects_analyzed}</p>
                                </div>
                            </div>
                        </Card>
                        <Card className="glass-card p-4">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-full bg-purple-500/20">
                                    <GitBranch className="w-5 h-5 text-purple-500" />
                                </div>
                                <div>
                                    <p className="text-sm text-muted-foreground">Clusters Found</p>
                                    <p className="text-2xl font-bold">{result.num_clusters}</p>
                                </div>
                            </div>
                        </Card>
                        <Card className="glass-card p-4">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-full bg-orange-500/20">
                                    <AlertCircle className="w-5 h-5 text-orange-500" />
                                </div>
                                <div>
                                    <p className="text-sm text-muted-foreground">Noise Points</p>
                                    <p className="text-2xl font-bold">{result.noise_points}</p>
                                </div>
                            </div>
                        </Card>
                        <Card className="glass-card p-4">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-full bg-green-500/20">
                                    <Lightbulb className="w-5 h-5 text-green-500" />
                                </div>
                                <div>
                                    <p className="text-sm text-muted-foreground">Cluster Rate</p>
                                    <p className="text-2xl font-bold">
                                        {formatPercent((result.total_defects_analyzed - result.noise_points) / result.total_defects_analyzed || 0)}
                                    </p>
                                </div>
                            </div>
                        </Card>
                    </div>

                    {/* Clusters */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {result.clusters.map((cluster) => (
                            <ClusterCard key={cluster.cluster_id} cluster={cluster} />
                        ))}
                    </div>

                    {/* Recommendations */}
                    {result.recommendations.length > 0 && (
                        <Card className="glass-card border-l-4 border-l-primary">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Lightbulb className="w-5 h-5" />
                                    Recommendations
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <ul className="space-y-2">
                                    {result.recommendations.map((rec, idx) => (
                                        <li key={idx} className="flex items-start gap-2">
                                            <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                                            <span className="text-sm text-muted-foreground">{rec}</span>
                                        </li>
                                    ))}
                                </ul>
                            </CardContent>
                        </Card>
                    )}
                </>
            )}

            {/* Empty State */}
            {!result && !loading && (
                <Card className="glass-card">
                    <CardContent className="p-12 text-center">
                        <GitBranch className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                        <h3 className="text-lg font-medium mb-2">No Analysis Yet</h3>
                        <p className="text-muted-foreground mb-4">
                            Configure the parameters above and run an analysis to find defect patterns
                        </p>
                    </CardContent>
                </Card>
            )}
        </main>
    )
}

function ClusterCard({ cluster }: { cluster: ClusterInfo }) {
    const severityColors: Record<string, string> = {
        critical: 'border-l-severity-critical',
        high: 'border-l-severity-high',
        medium: 'border-l-severity-medium',
        low: 'border-l-severity-low',
    }

    return (
        <Card className={`glass-card border-l-4 ${severityColors[cluster.avg_severity > 0.7 ? 'high' : cluster.avg_severity > 0.4 ? 'medium' : 'low']}`}>
            <CardHeader>
                <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">Cluster #{cluster.cluster_id}</CardTitle>
                    <Badge variant="secondary">{cluster.size} defects</Badge>
                </div>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                        <p className="text-muted-foreground">Dominant Type</p>
                        <p className="font-medium capitalize">{cluster.dominant_type.replace('_', ' ')}</p>
                    </div>
                    <div>
                        <p className="text-muted-foreground">Avg Severity</p>
                        <p className="font-medium">{formatPercent(cluster.avg_severity)}</p>
                    </div>
                    <div>
                        <p className="text-muted-foreground">Confidence</p>
                        <p className="font-medium">{formatPercent(cluster.confidence)}</p>
                    </div>
                    <div>
                        <p className="text-muted-foreground">Time Range</p>
                        <p className="font-medium text-xs">
                            {formatDateTime(cluster.time_range.start)} - {formatDateTime(cluster.time_range.end)}
                        </p>
                    </div>
                </div>

                <div className="p-3 rounded-lg bg-muted/50">
                    <p className="text-sm font-medium mb-1">Potential Cause</p>
                    <p className="text-sm text-muted-foreground">{cluster.potential_cause}</p>
                </div>
            </CardContent>
        </Card>
    )
}
