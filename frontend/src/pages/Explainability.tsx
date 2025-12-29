import { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { explainDetection, getXAIMethods } from '@/lib/api'
import { formatPercent, downloadBase64Image } from '@/lib/utils'
import type { XAIResponse } from '@/lib/types'
import {
    Brain,
    Upload,
    Download,
    RefreshCw,
    Eye,
    Sparkles,
    Info,
    Target,
} from 'lucide-react'

export default function Explainability() {
    const [file, setFile] = useState<File | null>(null)
    const [preview, setPreview] = useState<string | null>(null)
    const [result, setResult] = useState<XAIResponse | null>(null)
    const [loading, setLoading] = useState(false)
    const [method, setMethod] = useState('gradcam')
    const [viewMode, setViewMode] = useState<'original' | 'heatmap' | 'overlay'>('overlay')

    const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0]
        if (selectedFile) {
            setFile(selectedFile)
            setResult(null)
            const reader = new FileReader()
            reader.onload = (e) => setPreview(e.target?.result as string)
            reader.readAsDataURL(selectedFile)
        }
    }, [])

    const handleExplain = async () => {
        if (!file) return
        setLoading(true)
        try {
            const data = await explainDetection(file, {
                method,
                returnOverlay: true,
            })
            setResult(data)
        } catch (error) {
            console.error('Explanation failed:', error)
        } finally {
            setLoading(false)
        }
    }

    const getDisplayImage = () => {
        if (viewMode === 'heatmap' && result?.heatmap_base64) {
            return result.heatmap_base64
        }
        if (viewMode === 'overlay' && result?.overlay_base64) {
            return result.overlay_base64
        }
        return preview
    }

    return (
        <main className="p-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left: Upload and Preview */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Method Selection */}
                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Sparkles className="w-5 h-5" />
                                Explanation Method
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-3 gap-4">
                                <MethodCard
                                    title="Grad-CAM"
                                    description="Gradient-weighted activation maps"
                                    icon={<Eye className="w-5 h-5" />}
                                    selected={method === 'gradcam'}
                                    onClick={() => setMethod('gradcam')}
                                />
                                <MethodCard
                                    title="Occlusion"
                                    description="Sensitivity analysis"
                                    icon={<Target className="w-5 h-5" />}
                                    selected={method === 'occlusion'}
                                    onClick={() => setMethod('occlusion')}
                                />
                                <MethodCard
                                    title="Attention"
                                    description="Model attention weights"
                                    icon={<Brain className="w-5 h-5" />}
                                    selected={method === 'attention'}
                                    onClick={() => setMethod('attention')}
                                />
                            </div>
                        </CardContent>
                    </Card>

                    {/* Upload/Preview */}
                    {!preview ? (
                        <Card className="glass-card border-dashed border-2 cursor-pointer hover:border-primary/50 transition-colors">
                            <CardContent className="p-12">
                                <label className="flex flex-col items-center justify-center cursor-pointer">
                                    <div className="p-4 rounded-full bg-primary/10 mb-4">
                                        <Upload className="w-8 h-8 text-primary" />
                                    </div>
                                    <p className="text-lg font-medium">Upload an image for explanation</p>
                                    <p className="text-sm text-muted-foreground mt-2">
                                        See which regions the AI focuses on
                                    </p>
                                    <input
                                        type="file"
                                        accept="image/*"
                                        className="hidden"
                                        onChange={handleFileSelect}
                                    />
                                </label>
                            </CardContent>
                        </Card>
                    ) : (
                        <Card className="glass-card">
                            <CardHeader className="flex flex-row items-center justify-between">
                                <CardTitle className="text-lg">Visualization</CardTitle>
                                <div className="flex items-center gap-2">
                                    {result && (
                                        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as typeof viewMode)}>
                                            <TabsList>
                                                <TabsTrigger value="original">Original</TabsTrigger>
                                                <TabsTrigger value="overlay">Overlay</TabsTrigger>
                                                <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
                                            </TabsList>
                                        </Tabs>
                                    )}
                                    {result && (
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            onClick={() => downloadBase64Image(result.overlay_base64 || result.heatmap_base64, 'explanation.png')}
                                        >
                                            <Download className="w-4 h-4" />
                                        </Button>
                                    )}
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="relative bg-muted rounded-lg overflow-hidden">
                                    <img
                                        src={getDisplayImage() || ''}
                                        alt="Explanation visualization"
                                        className="w-full h-auto max-h-[450px] object-contain"
                                    />
                                </div>

                                <div className="flex gap-3 mt-4">
                                    <Button onClick={handleExplain} disabled={loading} className="flex-1">
                                        {loading ? (
                                            <>
                                                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                                                Generating Explanation...
                                            </>
                                        ) : (
                                            <>
                                                <Brain className="w-4 h-4 mr-2" />
                                                Generate Explanation
                                            </>
                                        )}
                                    </Button>
                                    <Button
                                        variant="outline"
                                        onClick={() => {
                                            setFile(null)
                                            setPreview(null)
                                            setResult(null)
                                        }}
                                    >
                                        Clear
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </div>

                {/* Right: Explanation Details */}
                <div className="space-y-6">
                    {/* Method Info */}
                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-lg">
                                <Info className="w-5 h-5" />
                                About {method === 'gradcam' ? 'Grad-CAM' : method === 'occlusion' ? 'Occlusion' : 'Attention'}
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-sm text-muted-foreground">
                                {method === 'gradcam' &&
                                    'Grad-CAM uses gradients flowing into the final convolutional layer to produce a coarse localization map highlighting important regions for the anomaly detection decision.'
                                }
                                {method === 'occlusion' &&
                                    'Occlusion sensitivity systematically occludes parts of the image to identify which regions are most important for the detection output.'
                                }
                                {method === 'attention' &&
                                    'Attention visualization shows which parts of the image the model attends to when making its decision, based on internal attention weights.'
                                }
                            </p>
                        </CardContent>
                    </Card>

                    {/* Explanation Text */}
                    {result?.explanation_text && (
                        <Card className="glass-card">
                            <CardHeader>
                                <CardTitle className="text-lg">Analysis Explanation</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <p className="text-sm text-muted-foreground leading-relaxed">
                                    {result.explanation_text}
                                </p>
                            </CardContent>
                        </Card>
                    )}

                    {/* Feature Importance */}
                    {result?.feature_importance && Object.keys(result.feature_importance).length > 0 && (
                        <Card className="glass-card">
                            <CardHeader>
                                <CardTitle className="text-lg">Feature Importance</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                {Object.entries(result.feature_importance)
                                    .sort(([, a], [, b]) => b - a)
                                    .map(([feature, importance]) => (
                                        <div key={feature}>
                                            <div className="flex justify-between text-sm mb-1">
                                                <span className="capitalize">{feature.replace('_', ' ')}</span>
                                                <span className="font-medium">{formatPercent(importance)}</span>
                                            </div>
                                            <div className="h-2 bg-muted rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-primary transition-all"
                                                    style={{ width: `${importance * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                    ))
                                }
                            </CardContent>
                        </Card>
                    )}

                    {/* Attention Regions */}
                    {result?.attention_regions?.length > 0 && (
                        <Card className="glass-card">
                            <CardHeader>
                                <CardTitle className="text-lg">Attention Regions</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <p className="text-sm text-muted-foreground">
                                    {result.attention_regions.length} region(s) of high attention detected
                                </p>
                            </CardContent>
                        </Card>
                    )}
                </div>
            </div>
        </main>
    )
}

function MethodCard({
    title,
    description,
    icon,
    selected,
    onClick,
}: {
    title: string
    description: string
    icon: React.ReactNode
    selected: boolean
    onClick: () => void
}) {
    return (
        <button
            onClick={onClick}
            className={`p-4 rounded-lg border-2 text-left transition-all ${selected
                    ? 'border-primary bg-primary/10'
                    : 'border-border hover:border-primary/50'
                }`}
        >
            <div className={`mb-2 ${selected ? 'text-primary' : 'text-muted-foreground'}`}>
                {icon}
            </div>
            <h4 className="font-medium">{title}</h4>
            <p className="text-xs text-muted-foreground mt-1">{description}</p>
        </button>
    )
}
