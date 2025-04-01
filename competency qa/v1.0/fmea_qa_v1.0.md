# Heidi Honours - FMEA competency questions

Version 1.0, 26/03/2025

### Simple

Q: What is the RPN for condenser
A: 18.

Q: What is the subsystem for the component engine
A: Power Unit

Q: What is the recommended action for wearing on the transmission mount?
A: Check bolt connection torques on 500 hour service

Q: What are the effects of leaking transmission hydraulics?
A: Loss of hydraulic oil

### Simple w/ Condition

Q: Which subcomponent of Power unit has an RPN of 27.
A: Power Unit>Air Intake System>Turbocharger
Note: there are no others with RPN of 27

Q: What is the one failure mode from the air intake system that still has a recommended action
A: Power unit > Air Intake System > Air filter > "Reaches end of usable life" (recommended to "Check air filter indicators on daily inspections")

Q: What could be the cause of the ECU overloading in the power train?
A: Shorting of sensor power surge
Note: The ECU sub-component also exists on the power unit and electrical system, and they also have overloading failure modes from different causes

### Set

Q. What are the main components of an airconditioner?
A. Compressor, Condenser, Evaporator, Fan, Filter, Filter Housing, Vents

Q. What are the RPNs for failure modes associated with the subcomponent Batteries?
A. There are three failure modes for the subcomponent Batteries. Corrosion (RPN 18), Wearing (RPN 8), and End of usable life (RPN 12).
Note: the regex 'batteries' also appears in the unstructured text of the Failure Effect column for the Alternator subcomponent

Q: What components have an RPN value over 35
A: Transmission (3 occurrences - ECU, ECU again, Planetary Gear Sets), Cooling System (Water Pump), Upbox (Gears), Dropbox (Gears).

Q: What components have an RPN value over 40
A: None

Q: What components (on which subsystems) have the failure mode Blocked?
A: Fuel filter (fuel system), oil cooler (lubrication system), oil filter (lubrication system), lubrication lines (lubrication systems) and filter (air conditioning).
There is also a Blockage failure mode on Fittings (Filters), and Blockages (plural) on Radiator (Cooling Systems) which has 2 identical entries, one with an action and one without.

### Comparison

Q: Which has a higher RPN value, the Engine or the Power Frame
A: It depends on the failure modes. The Engine's failure modes range from 16 to 32. The power frame's failure modes range from 2 to 24. In most causes the Engine's failure modes have a higher RPN, but this is not always the case.

Q: What component has the highest occurrence value for failure modes associated with leaks?
A: Power Unit > Cooling system > Radiator, with a failure mode named "Leaks" and an occurrence value of 4.
Note: Included in the comparison are failure modes with the names 'leak', 'leaks', 'leaking', or 'cracked and leaking'. There are also failure effects with the regex for 'leak'. Either way this answer is true

Q: What is the failure mode most likely to occur in the power train?
A: Power train > Wheelends > Tyre > Balding (occurence value of 4)
Note: This is the only power train failure mode with an occurence value of 4

### Aggregation

Q: How many distinct sub-components are listed in the data
A: 158 sub-components

Q: What is the average RPN of all failure modes for the charging and ignition component of the electrical system
A: Average of 20, over 14 different failure modes

Q: What is the average detection value over the full dataset?
A: 2.45

### Multi-hop

Q: What is the subsystem with the largest count of failure modes, that each have the highest RPN value in the dataset?
A: The highest RPN value in the dataset is 36, for failure modes of the following components:

-   Power unit > Cooling system
-   Power train > Upbox
-   Power train > Dropbox
-   Power train > Transmission (x3)
    Therefore the answer is the power train.

Q: There is a fire in the electrical system, what is the most likely reason for this?
A: There are two different electrical systems in the dataset - one is a subsystem and the other is a component of the power unit subsystem. Only the component electrical system has failure modes that have fire as a possible failure effect. The most likely ones of these have an occurence value of 3, caused by:

-   Shorting of connections / incorrect rating of fuse installed (Fuses > Overloading)
-   Exposure of the wiring harness to contaminants over time (Wiring harness > Corrosion)

Q: What is the most severe failure mode for the component with the highest number of failures?
A: The component with the highest number of failures is Electrical system > DCU (41 failure modes), and every one of them has a severity of 2

### Post-processing-heavy

Q: Are there any high RPN (25+) failure modes that are hard to detect (2-) that do not yet have a recommended action?
A: 2 failure modes that fit the description:

-   Power Unit > Engine > Belts > Mechanxical failure (RPN of 32, detection of 2, no recommended action)
-   Fire suppression system > AFFF > Activation system and gauge > Out of date / no test tag (RPN of 32, detection of 2, no recommended action)

Q: What proportion of failure causes + effects in the dataset have recommended actions to mitigate?
A: 0.58 = 192 / 329 (number of entries in dataset - 1, note that there is a single duplicate for failure mode + failure causes combinations in Power Unit > Cooling System > Radiator > Blockages)

Q: What percentage of hydraulic system failure modes have an RPN of above 15.
A: 43% = 12/28

Q: Are there any failure modes that have multiple causes?
A: **!! Can't easily get that from current data format, some 'multiple causes' are in multiple rows, some are concatencated in "possible causes of failure" column**

### False Premise

Q: What is the RPN for the Fuel System failure mode caused by software bugs?
A: No such failure mode for the Fuel System exists

Q: What is the effect of blockage of the hydraulic system valves fittings
A: The hydraulic system valves do not have a failure mode for blockage (although the hydraulic system filters do)

Q: There has been wearing on the fan drive assembly in the power unit. Is it more likely to be caused by physical impact or overheating?
A: Unknown. The fan drive assembly (Power unit > Cooling system > Fan drive assembly) only has one failure mode (wearing), with the cause "wearing over time". Note that this failure mode could lead to heating of the power unit (as an effect, not the other way around)

Q: What is the recommended action for corrosion of the Cabin controls wiper motor
A: No actions have been recommended for this failure mode
