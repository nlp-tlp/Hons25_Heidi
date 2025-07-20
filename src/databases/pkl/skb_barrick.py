from pydantic import Field
import csv

from .skb import SKB, SKBSchema, SKBNode

class BarrickSchema(SKBSchema):
    class Source(SKBNode):
        spreadsheet: str = Field(..., id=True)
        index: int = Field(..., id=True)

    class Subsystem(SKBNode):
        name: str = Field(..., id=True)

    class Component(SKBNode):
        part_of: list[str] = Field(..., id=True, relation=True, dest='Subsystem')
        name: str = Field(..., id=True)

    class SubComponent(SKBNode):
        part_of: list[str] = Field(..., id=True, relation=True, dest='Component')
        name: str = Field(..., id=True)

    class FailureMode(SKBNode):
        in_source: list[str] = Field(..., relation=True, dest='Source')
        for_part: list[str] = Field(..., id=True, relation=True, dest='SubComponent, Component, Subsystem')
        related_to: list[str] = Field(..., relation=True, dest='FailureCause, FailureEffect')
        has_action: list[str] = Field(..., relation=True, dest='CurrentControls, RecommendedAction')
        description: str = Field(..., id=True, semantic=True)
        occurrence: int = Field(..., id=True)
        detection: int = Field(..., id=True)
        rpn: int = Field(..., id=True)
        severity: int = Field(..., id=True)

    class FailureEffect(SKBNode):
        description: str = Field(..., id=True, semantic=True)

    class FailureCause(SKBNode):
        description: str = Field(..., id=True, semantic=True)

    class RecommendedAction(SKBNode):
        description: str = Field(..., id=True, semantic=True)

    class CurrentControls(SKBNode):
        description: str = Field(..., id=True, semantic=True)

def load_from_barrick_csv(skb: SKB, filepath: str, max_rows: int = None):
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            source = BarrickSchema.Source(spreadsheet=row["Spreadsheet"].strip(), index=i)
            source_id = skb.add_entity(source)

            subsystem = BarrickSchema.Subsystem(name=row["Subsystem"].strip())
            subsystem_id = skb.add_entity(subsystem)

            component = BarrickSchema.Component(part_of=[subsystem_id], name=row["Component"].strip())
            component_id = skb.add_entity(component)
            skb.get_entity_by_id(subsystem_id)._rev_in_subsystem = component_id

            subcomponent = BarrickSchema.SubComponent(part_of=[component_id], name=row["Sub-Component"].strip())
            subcomponent_id = skb.add_entity(subcomponent)

            fe = BarrickSchema.FailureEffect(description=row["Potential Effect(s) of Failure"].strip())
            fe_id = skb.add_entity(fe)

            fc = BarrickSchema.FailureCause(description=row["Potential Cause(s) of Failure"].strip())
            fc_id = skb.add_entity(fc)

            actions = []
            controls_str = row["Current Controls"]
            if controls_str:
                controls = BarrickSchema.CurrentControls(description=controls_str.strip())
                controls_id = skb.add_entity(controls)
                actions.append(controls_id)

            recommended_str = row["Recommended Action"]
            if recommended_str:
                recommended = BarrickSchema.RecommendedAction(description=recommended_str.strip())
                recommended_id = skb.add_entity(recommended)
                actions.append(recommended_id)

            fm = BarrickSchema.FailureMode(
                in_source=[source_id],
                for_part=[subcomponent_id],
                related_to=[fe_id, fc_id],
                has_action=actions,
                description=row["Potential Failure Mode"].strip(),
                occurrence=int(row["Occurrence"]),
                detection=int(row["Detection"]),
                rpn=int(row["RPN"]),
                severity=int(row["Severity"])
            )
            skb.add_entity(fm)

    return skb
