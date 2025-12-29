import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { getSettings, updateSettings, getHealth } from '@/lib/api'
import type { SettingsResponse, HealthResponse } from '@/lib/types'
import {
    Settings as SettingsIcon,
    Save,
    RefreshCw,
    Cpu,
    Gauge,
    Sliders,
    Shield,
    CheckCircle,
    XCircle,
} from 'lucide-react'

export default function Settings() {
    const [settings, setSettings] = useState<SettingsResponse | null>(null)
    const [health, setHealth] = useState<HealthResponse | null>(null)
    const [loading, setLoading] = useState(true)
    const [saving, setSaving] = useState(false)
    const [modified, setModified] = useState(false)

    // Local form state
    const [detectionThreshold, setDetectionThreshold] = useState(0.5)
    const [localizationThreshold, setLocalizationThreshold] = useState(0.3)
    const [areaWeight, setAreaWeight] = useState(0.4)
    const [intensityWeight, setIntensityWeight] = useState(0.3)
    const [locationWeight, setLocationWeight] = useState(0.3)

    useEffect(() => {
        fetchData()
    }, [])

    const fetchData = async () => {
        setLoading(true)
        try {
            const [settingsData, healthData] = await Promise.all([
                getSettings(),
                getHealth(),
            ])
            setSettings(settingsData)
            setHealth(healthData)

            // Initialize form state
            setDetectionThreshold(settingsData.detection_threshold)
            setLocalizationThreshold(settingsData.localization_threshold)
            setAreaWeight(settingsData.severity_weights.area)
            setIntensityWeight(settingsData.severity_weights.intensity)
            setLocationWeight(settingsData.severity_weights.location)
        } catch (error) {
            console.error('Failed to fetch settings:', error)
        } finally {
            setLoading(false)
        }
    }

    const handleSave = async () => {
        setSaving(true)
        try {
            const updated = await updateSettings({
                detectionThreshold,
                localizationThreshold,
                severityWeights: {
                    area: areaWeight,
                    intensity: intensityWeight,
                    location: locationWeight,
                },
            })
            setSettings(updated)
            setModified(false)
        } catch (error) {
            console.error('Failed to save settings:', error)
        } finally {
            setSaving(false)
        }
    }

    const handleChange = () => {
        setModified(true)
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
            </div>
        )
    }

    return (
        <main className="p-6 space-y-6">
            {/* System Status */}
            <Card className="glass-card">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Cpu className="w-5 h-5" />
                        System Status
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <StatusItem
                            label="API Status"
                            value={health?.status || 'Unknown'}
                            status={health?.status === 'healthy'}
                        />
                        <StatusItem
                            label="Model Loaded"
                            value={health?.model_loaded ? 'Yes' : 'No'}
                            status={health?.model_loaded}
                        />
                        <StatusItem
                            label="GPU Available"
                            value={health?.gpu_available ? 'Yes' : 'No'}
                            status={health?.gpu_available}
                        />
                        <StatusItem
                            label="Device"
                            value={health?.device || 'Unknown'}
                            status={true}
                        />
                    </div>
                    {health?.gpu_name && (
                        <div className="mt-4 p-3 rounded-lg bg-muted">
                            <p className="text-sm text-muted-foreground">
                                GPU: <span className="font-medium text-foreground">{health.gpu_name}</span>
                            </p>
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* Detection Settings */}
            <Card className="glass-card">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Gauge className="w-5 h-5" />
                        Detection Thresholds
                    </CardTitle>
                    <CardDescription>
                        Adjust sensitivity for defect detection
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="text-sm font-medium">Detection Threshold</label>
                            <span className="text-sm font-mono">{(detectionThreshold * 100).toFixed(0)}%</span>
                        </div>
                        <Slider
                            value={[detectionThreshold * 100]}
                            onValueChange={([v]) => {
                                setDetectionThreshold(v / 100)
                                handleChange()
                            }}
                            min={0}
                            max={100}
                            step={1}
                        />
                        <p className="text-xs text-muted-foreground">
                            Higher values reduce false positives but may miss subtle defects
                        </p>
                    </div>

                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="text-sm font-medium">Localization Threshold</label>
                            <span className="text-sm font-mono">{(localizationThreshold * 100).toFixed(0)}%</span>
                        </div>
                        <Slider
                            value={[localizationThreshold * 100]}
                            onValueChange={([v]) => {
                                setLocalizationThreshold(v / 100)
                                handleChange()
                            }}
                            min={0}
                            max={100}
                            step={1}
                        />
                        <p className="text-xs text-muted-foreground">
                            Controls sensitivity for pixel-level defect localization
                        </p>
                    </div>
                </CardContent>
            </Card>

            {/* Severity Weights */}
            <Card className="glass-card">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Sliders className="w-5 h-5" />
                        Severity Scoring Weights
                    </CardTitle>
                    <CardDescription>
                        Adjust how different factors contribute to severity scores
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="text-sm font-medium">Area Weight</label>
                            <span className="text-sm font-mono">{(areaWeight * 100).toFixed(0)}%</span>
                        </div>
                        <Slider
                            value={[areaWeight * 100]}
                            onValueChange={([v]) => {
                                setAreaWeight(v / 100)
                                handleChange()
                            }}
                            min={0}
                            max={100}
                            step={5}
                        />
                    </div>

                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="text-sm font-medium">Intensity Weight</label>
                            <span className="text-sm font-mono">{(intensityWeight * 100).toFixed(0)}%</span>
                        </div>
                        <Slider
                            value={[intensityWeight * 100]}
                            onValueChange={([v]) => {
                                setIntensityWeight(v / 100)
                                handleChange()
                            }}
                            min={0}
                            max={100}
                            step={5}
                        />
                    </div>

                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="text-sm font-medium">Location Weight</label>
                            <span className="text-sm font-mono">{(locationWeight * 100).toFixed(0)}%</span>
                        </div>
                        <Slider
                            value={[locationWeight * 100]}
                            onValueChange={([v]) => {
                                setLocationWeight(v / 100)
                                handleChange()
                            }}
                            min={0}
                            max={100}
                            step={5}
                        />
                    </div>

                    <div className="p-3 rounded-lg bg-muted">
                        <p className="text-sm text-muted-foreground">
                            Total weight: {((areaWeight + intensityWeight + locationWeight) * 100).toFixed(0)}%
                            {Math.abs(areaWeight + intensityWeight + locationWeight - 1) > 0.01 && (
                                <span className="text-yellow-500 ml-2">(will be normalized)</span>
                            )}
                        </p>
                    </div>
                </CardContent>
            </Card>

            {/* System Info */}
            <Card className="glass-card">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Shield className="w-5 h-5" />
                        System Information
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                            <p className="text-muted-foreground">Version</p>
                            <p className="font-medium">{health?.version || '1.0.0'}</p>
                        </div>
                        <div>
                            <p className="text-muted-foreground">Max Batch Size</p>
                            <p className="font-medium">{settings?.max_batch_size || 32}</p>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Save Button */}
            <div className="flex justify-end gap-3">
                <Button variant="outline" onClick={fetchData}>
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Reset
                </Button>
                <Button onClick={handleSave} disabled={saving || !modified}>
                    {saving ? (
                        <>
                            <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                            Saving...
                        </>
                    ) : (
                        <>
                            <Save className="w-4 h-4 mr-2" />
                            Save Changes
                        </>
                    )}
                </Button>
            </div>
        </main>
    )
}

function StatusItem({
    label,
    value,
    status,
}: {
    label: string
    value: string
    status?: boolean
}) {
    return (
        <div className="p-3 rounded-lg bg-muted">
            <div className="flex items-center gap-2">
                {status !== undefined && (
                    status ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                        <XCircle className="w-4 h-4 text-red-500" />
                    )
                )}
                <p className="text-sm text-muted-foreground">{label}</p>
            </div>
            <p className="font-medium mt-1 capitalize">{value}</p>
        </div>
    )
}
